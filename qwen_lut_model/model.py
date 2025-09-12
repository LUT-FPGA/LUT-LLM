import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os
import glob

def load_config(config_path):
    """
    Load configuration from a JSON file.
    TODO: Implement loading of language model configuration and FPGA hardware configuration.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def max_parallel_sub_per_dsp(dsp_type='dsp58', act_bit=32):
    # TODO: use fpga resource config
    if dsp_type == 'dsp58':
        if act_bit <= 8:
            return 6
        elif act_bit <= 16:
            return 3
        elif act_bit <= 28:
            return 2
        else:
            return 1
            
def LUTLinear(
    vec_len,
    n_centroids,
    dim_in,
    dim_out,
    seq_len,
    max_lutl,
    max_mem,
    max_bank,
    data_bw,
    parallel_acc,
    dsp_per_float=1,
    fixed=False,
    CCU_reload_factor=1,
    act_bit=32,
    lut_bit=32,
    decoding=False,
    debug=False
):
    if decoding:
        seq_len = 1

    dsp = 0
    memory = 0
    lut = 0
    latency = 0

    sub_per_dsp = max_parallel_sub_per_dsp("dsp58", act_bit)

    dPE_dsp = vec_len * dsp_per_float # each for fp32 fsub
    dPE_lut = act_bit * (vec_len + 1) // 2
    dPE_lat = 1 + np.log2(vec_len)
    CCU_dsp = dPE_dsp * n_centroids * (data_bw // (vec_len * act_bit // 8)) / CCU_reload_factor / sub_per_dsp
    CCU_lut = dPE_lut * n_centroids * (data_bw // (vec_len * act_bit // 8)) / CCU_reload_factor
    CCU_lat = (dPE_lat * np.log2(n_centroids) + seq_len) * (dim_in * (act_bit // 8) / data_bw) * CCU_reload_factor # pipelined per CCU
    centroid_buffer_mem = n_centroids * (act_bit//8) * vec_len * (dim_in//vec_len) # 4 bytes per float
    if not fixed:
        total_mem_required = n_centroids * lut_bit//8 * (dim_in//vec_len) * dim_out
        psum_lut_lat = 1
        if total_mem_required > (max_mem - centroid_buffer_mem):
            psum_lut_lat = total_mem_required / (max_mem - centroid_buffer_mem) # time to load lut
            total_mem_required = max_mem - centroid_buffer_mem
        gemm_lat = (dim_in * (act_bit // 8) / data_bw) * seq_len
        if psum_lut_lat > CCU_reload_factor:
            gemm_lat *= (psum_lut_lat // CCU_reload_factor)
        simutaneous_access = (data_bw // (vec_len * (lut_bit // 8))) * (n_centroids * (lut_bit // 8) * (dim_in//vec_len) // total_mem_required)
        if simutaneous_access > max_bank:
            psum_lut_lat = (simutaneous_access / max_bank) * gemm_lat
        else:
            psum_lut_lat = gemm_lat
        psum_lut_dsp = parallel_acc * dsp_per_float * (lut_bit / 32)
        # latency for accumulation can overlap with multiplication 
        dsp = CCU_dsp + psum_lut_dsp
        lut = CCU_lut
        latency = max(CCU_lat, psum_lut_lat) + dPE_lat * n_centroids
        if debug:
            print(f"CCU latency: {CCU_lat}, PSUM LUT latency: {psum_lut_lat}")
        memory = centroid_buffer_mem + total_mem_required / CCU_reload_factor * 7
    else:
        single_table_lut_cost = 32/3 * (2**(np.log2(n_centroids)) - (-1)**np.log2(n_centroids))
        lutl_required = (dim_in//vec_len)*single_table_lut_cost*dim_out
        if lutl_required > max_lutl:
            print("cannot implement this linear layer with pure SLICEL CLB")
            exit(1)
        gemm_lat = (dim_in * 4 / data_bw) * seq_len
        psum_lut_dsp = parallel_acc * dsp_per_float
        dsp = CCU_dsp + psum_lut_dsp
        lut = CCU_lut + lutl_required
        latency = CCU_lat + gemm_lat
        memory = centroid_buffer_mem

    input_size = dim_in * act_bit // 8 * seq_len
    output_size = dim_out * 4 * seq_len
    
    return dsp, lut, memory, latency, input_size, output_size

def GroupQueryQK(seq_len, head_dim, n_q_head, n_kv_head, pea_x, pea_y, pea_z, dsp_per_float=1, decoding=False):
    sub_len = seq_len
    if decoding:
        sub_len = 1
    total_mac_ops = seq_len * sub_len * head_dim * n_q_head
    total_PE = pea_x * pea_y * pea_z 
    latency = total_mac_ops * dsp_per_float / total_PE
    dsp = total_PE
    input_size = seq_len * head_dim * 4 * (n_q_head + 2 * n_kv_head)
    output_size = seq_len * sub_len * 4 * n_q_head
    return dsp, 0, 0, latency, input_size, output_size

def GroupQueryAV(seq_len, head_dim, n_q_head, pea_x, pea_y, pea_z, dsp_per_float=1, decoding=False):
    sub_len = seq_len
    if decoding:
        sub_len = 1
    total_mac_ops = seq_len * sub_len * head_dim * n_q_head
    total_PE = pea_x * pea_y * pea_z 
    latency = total_mac_ops * dsp_per_float / total_PE
    dsp = total_PE
    input_size = seq_len * sub_len * 4 * n_q_head
    output_size = seq_len * head_dim * 4 * n_q_head
    return dsp, 0, 0, latency, input_size, output_size

def RoPE(seq_len, head_dim, n_q_head, n_kv_head, dsp_per_float=1, decoding=False, parallel_factor=32):
    if decoding:
        seq_len = 1
    # get from hls report
    dsp = parallel_factor * dsp_per_float
    memory = parallel_factor * 18 * 1024 / 8
    latency = seq_len * head_dim * (n_q_head + n_kv_head) / dsp
    input_size = seq_len * head_dim * 4 * (n_q_head + n_kv_head)
    output_size = input_size
    return dsp, memory, 0, latency, input_size, output_size

def LayerNorm(seq_len, hidden_dim, dsp_per_float=1, decoding=False, parallel_factor=32):
    if decoding:
        seq_len = 1
    # get from hls report
    dsp = parallel_factor * 2 * dsp_per_float
    memory = 68 * (parallel_factor / 16) * 18 * 1024 / 8
    latency = seq_len * hidden_dim * 2 / dsp
    input_size = seq_len * hidden_dim * 4
    output_size = input_size
    return dsp, memory, 0, latency, input_size, output_size
 
def SwiGLU(seq_len, hidden_dim, dsp_per_float=1, decoding=False, parallel_factor=32):
    if decoding:
        seq_len = 1
    # SwiGLU(x) = Wx * sigmoid(Wx) .* (Vx)
    # exp(*): 3 BRAM
    # fmul, fadd: require 5 in total
    dsp = 5 * parallel_factor * dsp_per_float
    memory = (parallel_factor + parallel_factor * 3) * 18 * 1024 / 8
    latency = seq_len * hidden_dim / parallel_factor
    input_size = seq_len * hidden_dim * 4
    output_size = seq_len * hidden_dim * 4
    return dsp, memory, 0, latency, input_size, output_size


def fuse_op(op1_config: Tuple, op2_config: Tuple):
    return (
        op1_config[0] + op2_config[0],
        op1_config[1] + op2_config[1],
        op1_config[2] + op2_config[2],
        max(op1_config[3], op2_config[3]),
        op1_config[4],
        op2_config[5],
    )

def parallel_op(op1_config: Tuple, op2_config: Tuple):
    return (
        op1_config[0] + op2_config[0],
        op1_config[1] + op2_config[1],
        op1_config[2] + op2_config[2],
        max(op1_config[3], op2_config[3]),
        op1_config[4] + op2_config[4],
        op1_config[5] + op2_config[5],
    )

def parallel_op_list(op_list: List[Tuple], share_input=False, accumulate_output=False):
    return (
        sum(op[0] for op in op_list),
        sum(op[1] for op in op_list),
        sum(op[2] for op in op_list),
        max(op[3] for op in op_list),
        sum(op[4] for op in op_list) if not share_input else op_list[0][4],
        sum(op[5] for op in op_list) if not accumulate_output else op_list[-1][5],
    )

def overlay_op_list(op_list: List[Tuple]):
    return (
        max(op[0] for op in op_list),
        max(op[1] for op in op_list),
        max(op[2] for op in op_list),
        sum(op[3] for op in op_list),
        max(op[4] for op in op_list),
        max(op[5] for op in op_list),
    )


def QwenModel(
    seq_len, # sequence length of input
    hidden_dim, # hidden dimension
    intermediate_dim, # dimension of FFN layer
    head_dim, # head dimension
    n_q_head, # number of heads in total
    n_kv_head, # number of KV heads
    n_layer, # number of layers
    vec_len, # LUTLinear: vector length
    n_centroids, # LUTLinear: number of centroids per LUT
    pea_x, # GEMM: number of PE in x direction
    pea_y, # GEMM: number of PE in y direction
    pea_z, # GEMM: number of PE in z direction
    max_lutl, # device: max LUT in SLICEL CLB
    max_mem, # device: total PL memory
    max_bank, # device: number of memory banks (LUTRAM count 1 bank per LUT)
    parallel_acc, # LUTLinear: number of parallel accumulations
    off_chip_port, # device: off-chip port count
    dsp_per_float=1, # DSP per float
    CCU_reload_factor=1, # CCU reduce factor
    act_bit=32, # LUTLinear: bitwidth of activation
    lut_bit=32, # LUTLinear: bitwidth of LUT
    nonlinear_para_factor=32,
    decoding=False, # whether to decode
    weight_vq=False # whether to use weight VQ
):

    model_config = (0, 0, 0, 0, seq_len * hidden_dim * 4, 0)
    breakdown = {
        "qkv_proj": 0,
        "gqa_qk": 0,
        "gqa_av": 0,
        "out_proj": 0,
        "up_gate_proj": 0,
        "down_proj": 0
    }
    for layer in range(n_layer):
        model_sub_config = (0, 0, 0, 0, seq_len * hidden_dim * 4, 0)
        
        # 1. QKV Projection, fused with RoPE
        gqa_share_factor = n_q_head // n_kv_head + 2
        max_memory = max_mem / gqa_share_factor
        max_memory_bank = max_bank / gqa_share_factor

        data_bw_factor = 1.5 if weight_vq else 1
        
        q_proj_rope = fuse_op(
            LUTLinear(
                vec_len, n_centroids, hidden_dim, head_dim, seq_len, 
                max_lutl, max_memory, max_memory_bank, 
                512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc, dsp_per_float, 
                fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
                decoding=decoding
            ),
            RoPE(
                seq_len, head_dim, n_q_head, n_kv_head, dsp_per_float,
                decoding=decoding, parallel_factor=nonlinear_para_factor
            )
        )
        
        # parallelize q for every group
        group_q_proj = parallel_op_list([q_proj_rope] * (n_q_head // n_kv_head), share_input=True)
        
        k_proj_rope = fuse_op(
            LUTLinear(
                vec_len, n_centroids, hidden_dim, head_dim, seq_len, 
                max_lutl, max_memory, max_memory_bank, 
                512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc, dsp_per_float, 
                fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
                decoding=decoding
            ),
            RoPE(
                seq_len, head_dim, n_q_head, n_kv_head, dsp_per_float,
                decoding=decoding, parallel_factor=nonlinear_para_factor
            )
        )
        
        v_proj = LUTLinear(
            vec_len, n_centroids, hidden_dim, head_dim, seq_len, 
            max_lutl, max_memory, max_memory_bank, 
            512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc, dsp_per_float, 
            fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
            decoding=decoding
        )
        
        # 2. Group Query Attention
        gqa_qk = GroupQueryQK(
            seq_len, head_dim, n_q_head // n_kv_head, 1, pea_x, pea_y, pea_z, dsp_per_float,
            decoding=decoding
        )
        
        gqa_av = GroupQueryAV(
            seq_len, head_dim, n_q_head // n_kv_head, pea_x, pea_y, pea_z, dsp_per_float,
            decoding=decoding
        )
        
        out_proj = LUTLinear(
            vec_len, n_centroids, hidden_dim, hidden_dim, seq_len, 
            max_lutl, max_mem, max_bank, 
            512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc * gqa_share_factor, dsp_per_float, 
            fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
            decoding=decoding
        )

        ln = LayerNorm(seq_len, hidden_dim, dsp_per_float, decoding=decoding, parallel_factor=nonlinear_para_factor)

        for i in range(n_kv_head):
            model_sub_config = overlay_op_list([
                model_sub_config, 
                parallel_op_list([group_q_proj, k_proj_rope, v_proj], share_input=True), 
                fuse_op(gqa_qk, gqa_av)
            ])
            breakdown["qkv_proj"] += parallel_op_list([group_q_proj, k_proj_rope, v_proj])[3]
            breakdown["gqa_qk"] += gqa_qk[3]
            breakdown["gqa_av"] += gqa_av[3]
            
        model_sub_config = overlay_op_list([model_sub_config, fuse_op(out_proj, ln)])
        breakdown["out_proj"] += out_proj[3]

        # 3. FFN: SwiGLU(FFN1, FFN2) * FFN3
        ffn1 = LUTLinear(
            vec_len, n_centroids, hidden_dim, intermediate_dim, seq_len, 
            max_lutl/2, max_mem/2, max_bank/2, 
            512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc * gqa_share_factor/2, dsp_per_float, 
            fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
            decoding=decoding
        )
        
        ffn2 = LUTLinear(
            vec_len, n_centroids, hidden_dim, intermediate_dim, seq_len, 
            max_lutl/2, max_mem/2, max_bank/2, 
            512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc * gqa_share_factor/2, dsp_per_float, 
            fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
            decoding=decoding
        )
        
        up_gate_proj = fuse_op(
            parallel_op_list([ffn1, ffn2], share_input=True),
            SwiGLU(seq_len, intermediate_dim, dsp_per_float, decoding=decoding, parallel_factor=nonlinear_para_factor)
        )
        breakdown["up_gate_proj"] += up_gate_proj[3]
        
        ffn3 = LUTLinear(
            vec_len, n_centroids, intermediate_dim, hidden_dim, seq_len, 
            max_lutl, max_mem, max_bank, 
            512 * off_chip_port * 3 / 4 / 8 * data_bw_factor, parallel_acc * gqa_share_factor, dsp_per_float, 
            fixed=False, CCU_reload_factor=CCU_reload_factor, act_bit=act_bit, lut_bit=lut_bit,
            decoding=decoding
        )
        
        ln2 = LayerNorm(seq_len, hidden_dim, dsp_per_float, decoding=decoding, 
                       parallel_factor=nonlinear_para_factor)
        
        model_sub_config = overlay_op_list([model_sub_config, up_gate_proj, fuse_op(ffn3, ln2)])
        model_config = overlay_op_list([model_config, model_sub_config])
        breakdown["down_proj"] += fuse_op(ffn3, ln2)[3]

    return model_config, breakdown

def check_resources(model_estimate: Tuple, fpga_config: dict, verbose: bool = True) -> bool:
    """
    Check if the estimated resource usage fits within FPGA constraints.
    
    Args:
        model_estimate: Tuple containing (dsp, lut, memory, latency, input_size, output_size)
        fpga_config: Dictionary containing FPGA resource specifications
        verbose: Whether to print detailed resource usage information
    
    Returns:
        bool: True if resources fit within constraints, False otherwise
    """
    # Extract estimated usage
    est_dsp = model_estimate[0]
    est_lut = model_estimate[1] 
    est_memory_bytes = model_estimate[2] + model_estimate[4] + model_estimate[5]
    
    # Extract FPGA limits
    max_dsp = fpga_config['dsp']
    max_lut = fpga_config['lut'] // 2
    
    # Calculate total memory available (BRAM + URAM)
    # BRAM18K: 18Kb = 2304 bytes each
    # URAM: 288Kb = 36864 bytes each  
    bram_bytes = fpga_config['bram'] * 2304
    uram_bytes = fpga_config['uram'] * 36864
    lutram_bytes = fpga_config['lut'] // 2 * 8
    max_memory_bytes = bram_bytes + uram_bytes + lutram_bytes
    
    # Check resource utilization
    dsp_utilization = (est_dsp / max_dsp) * 100
    lut_utilization = (est_lut / max_lut) * 100
    memory_utilization = (est_memory_bytes / max_memory_bytes) * 100
    
    # Determine if resources fit
    resources_fit = (est_dsp <= max_dsp) and (est_lut <= max_lut) and (est_memory_bytes <= max_memory_bytes)
    spill_data_size = max(est_memory_bytes - max_memory_bytes, 0)
    
    if verbose:
        print("\n" + "="*60)
        print("FPGA RESOURCE UTILIZATION CHECK")
        print("="*60)
        
        # DSP check
        status_dsp = "PASS" if est_dsp <= max_dsp else "FAIL"
        print(f"DSP:         {est_dsp:8,.0f} / {max_dsp:8,.0f} ({dsp_utilization:5.1f}%) {status_dsp}")
        
        # LUT check  
        status_lut = "PASS" if est_lut <= max_lut else "FAIL"
        print(f"LUT for Logic: {est_lut:8,.0f} / {max_lut:8,.0f} ({lut_utilization:5.1f}%) {status_lut}")
        
        # Memory check
        status_mem = "PASS" if est_memory_bytes <= max_memory_bytes else "FAIL"
        print(f"Memory:         {est_memory_bytes:8,.0f} / {max_memory_bytes:8,.0f} bytes ({memory_utilization:5.1f}%) {status_mem}")
        
        print("-"*60)
        print(f"Memory breakdown:")
        print(f"  BRAM ({fpga_config['bram']} x 2304 bytes): {bram_bytes:12,.0f} bytes")
        print(f"  URAM ({fpga_config['uram']} x 36864 bytes): {uram_bytes:12,.0f} bytes")
        print(f"  LUTRAM ({fpga_config['lut'] // 2} x 8 bytes): {lutram_bytes:12,.0f} bytes")
        
        overall_status = "PASS - Model fits within FPGA constraints" if resources_fit else "FAIL - Model exceeds FPGA constraints"
        print(f"\nOverall Status: {overall_status}")
        print("="*60)
        
        if not resources_fit:
            print("\nWARNING: Resource constraints violated!")
            if est_dsp > max_dsp:
                print(f"   DSP shortage: Need {est_dsp - max_dsp:,.0f} additional DSP blocks")
            if est_lut > max_lut:
                print(f"   LUT shortage: Need {est_lut - max_lut:,.0f} additional LUTs")
            if est_memory_bytes > max_memory_bytes:
                print(f"   Memory shortage: Need {est_memory_bytes - max_memory_bytes:,.0f} additional bytes")
            print()
    
    return resources_fit, spill_data_size

def compute_model_operations(
    seq_len, # sequence length of input
    hidden_dim, # hidden dimension
    intermediate_dim, # dimension of FFN layer
    head_dim, # head dimension
    n_q_head, # number of heads in total
    n_kv_head, # number of KV heads
    n_layer, # number of layers
    decoding=False # whether to decode
):
    """
    Compute total number of operations (FLOPs) for model inference.
    
    Returns:
        dict: Dictionary containing operation counts for each component
        int: Total operations count
    """
    
    ops_breakdown = {
        "qkv_proj": 0,
        "rope": 0, 
        "attention_qk": 0,
        "attention_av": 0,
        "out_proj": 0,
        "ffn_up_gate": 0,
        "ffn_down": 0,
        "swiglu": 0,
        "layer_norm": 0
    }
    
    for layer in range(n_layer):
        # 1. QKV Projection
        if decoding:
        # Q projection: seq_len * hidden_dim * (head_dim * n_q_head) * 2 (multiply-add)
            q_proj_ops = hidden_dim * (head_dim * n_q_head) * 2
            
            # K projection: seq_len * hidden_dim * (head_dim * n_kv_head) * 2
            k_proj_ops = hidden_dim * (head_dim * n_kv_head) * 2
            
            # V projection: seq_len * hidden_dim * (head_dim * n_kv_head) * 2  
            v_proj_ops = hidden_dim * (head_dim * n_kv_head) * 2
        else:
            q_proj_ops = seq_len * hidden_dim * (head_dim * n_q_head) * 2
            k_proj_ops = seq_len * hidden_dim * (head_dim * n_kv_head) * 2
            v_proj_ops = seq_len * hidden_dim * (head_dim * n_kv_head) * 2
        
        ops_breakdown["qkv_proj"] += q_proj_ops + k_proj_ops + v_proj_ops
        
        # 2. RoPE (Rotary Position Embedding)
        # Each position requires 2 ops per dimension (sin/cos computation)
        # Applied to Q and K tensors
        if decoding:
            rope_ops = head_dim * (n_q_head + n_kv_head) * 2
        else:
            rope_ops = seq_len * head_dim * (n_q_head + n_kv_head) * 2
        ops_breakdown["rope"] += rope_ops
        
        # 3. Attention QK^T
        # For each head group: seq_len * seq_len * head_dim * 2 (multiply-add)
        if not decoding:
            qk_ops = seq_len * seq_len * head_dim * n_q_head * 2
        else:
            # During decoding, we only compute attention with past tokens
            qk_ops = seq_len * head_dim * n_q_head * 2
        ops_breakdown["attention_qk"] += qk_ops
        
        # 4. Attention AV  
        # For each head: seq_len * head_dim * seq_len * 2
        if not decoding:
            av_ops = seq_len * head_dim * seq_len * n_q_head * 2
        else:
            av_ops = seq_len * head_dim * n_q_head * 2
        ops_breakdown["attention_av"] += av_ops
        
        # 5. Output Projection
        # seq_len * (head_dim * n_q_head) * hidden_dim * 2
        if not decoding:
            out_proj_ops = seq_len * (head_dim * n_q_head) * hidden_dim * 2
        else:
            out_proj_ops = hidden_dim * (head_dim * n_q_head) * 2
        ops_breakdown["out_proj"] += out_proj_ops
        
        # 6. FFN Up/Gate Projection (parallel)
        # Up: seq_len * hidden_dim * intermediate_dim * 2
        # Gate: seq_len * hidden_dim * intermediate_dim * 2
        if not decoding:
            ffn_up_gate_ops = seq_len * hidden_dim * intermediate_dim * 2 * 2  # Two projections
        else:
            ffn_up_gate_ops = hidden_dim * intermediate_dim * 2 * 2  # Two projections
        ops_breakdown["ffn_up_gate"] += ffn_up_gate_ops
        
        # 7. SwiGLU Activation
        # sigmoid: ~4 ops per element, multiply: 1 op per element
        if not decoding:
            swiglu_ops = seq_len * intermediate_dim * 5
        else:
            swiglu_ops = intermediate_dim * 5
        ops_breakdown["swiglu"] += swiglu_ops
        
        # 8. FFN Down Projection  
        # seq_len * intermediate_dim * hidden_dim * 2
        if not decoding:
            ffn_down_ops = seq_len * intermediate_dim * hidden_dim * 2
        else:
            ffn_down_ops = intermediate_dim * hidden_dim * 2
        ops_breakdown["ffn_down"] += ffn_down_ops
        
        # 9. Layer Normalization (2 per layer: before attention, before FFN)
        # Each LayerNorm: mean (1 op per element), variance (2 ops per element), 
        # normalize (3 ops per element) = 6 ops per element
        if not decoding:
            layer_norm_ops = seq_len * hidden_dim * 6 * 2  # Two layer norms per layer
        else:
            layer_norm_ops = hidden_dim * 6 * 2  # Two layer norms per layer
        ops_breakdown["layer_norm"] += layer_norm_ops
    
    total_ops = sum(ops_breakdown.values())
    
    return ops_breakdown, total_ops

def generate_roofline_model(model_config, fpga_config, hyperparams):
    """
    Generate roofline model data by sampling different configurations.
    
    Returns:
        Lists of operational intensities, throughputs, and labels for plotting
    """
    # Sample different sequence lengths
    seq_lengths = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    decoding_modes = [True, False]
    
    operational_intensities = []
    throughputs = []
    labels = []
    
    # Load tech config for DSP per float
    tech_config = load_config('fpga_resource_config.json')
    dsp_per_float = 1
    for dsp_conf in tech_config['dsp']:
        if dsp_conf['version'] == fpga_config['dsp_version']:
            dsp_per_float = dsp_conf['operations'][0]['count']
            break
    
    bram_bytes = fpga_config['bram'] * 2304
    uram_bytes = fpga_config['uram'] * 36864
    lutram_bytes = fpga_config['lut'] // 2 * 8
    max_memory_bytes = bram_bytes + uram_bytes + lutram_bytes
    
    for seq_len in seq_lengths:
        for decoding in decoding_modes:
            try:
                # Compute operations
                _, total_ops = compute_model_operations(
                    seq_len=seq_len,
                    hidden_dim=model_config['hidden_size'],
                    intermediate_dim=model_config['intermediate_size'],
                    head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                    n_q_head=model_config['num_attention_heads'],
                    n_kv_head=model_config['num_key_value_heads'],
                    n_layer=model_config['num_hidden_layers'],
                    decoding=decoding
                )
                
                # Compute model estimate
                model_estimate, _ = QwenModel(
                    seq_len=seq_len,
                    hidden_dim=model_config['hidden_size'],
                    intermediate_dim=model_config['intermediate_size'],
                    head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                    n_q_head=model_config['num_attention_heads'],
                    n_kv_head=model_config['num_key_value_heads'],
                    n_layer=model_config['num_hidden_layers'],
                    vec_len=hyperparams['vec_len'],
                    n_centroids=hyperparams['n_centroids'],
                    pea_x=hyperparams['pea_x'],
                    pea_y=hyperparams['pea_y'],
                    pea_z=hyperparams['pea_z'],
                    max_lutl=fpga_config['lut']//2,
                    max_mem=max_memory_bytes,
                    max_bank=fpga_config['bram']+fpga_config['uram']+256,
                    parallel_acc=hyperparams['parallel_acc'],
                    off_chip_port=fpga_config['off_chip_port'],
                    dsp_per_float=dsp_per_float,
                    CCU_reload_factor=hyperparams['CCU_reload_factor'],
                    act_bit=hyperparams['act_bit'],
                    lut_bit=hyperparams['lut_bit'],
                    nonlinear_para_factor=hyperparams['nonlinear_para_factor'],
                    decoding=decoding,
                    weight_vq=hyperparams['weight_vq']
                )
                
                # Calculate memory access (approximate)
                # Calculate memory access (approximate)
                if "weight_vq" not in hyperparams or not hyperparams['weight_vq']:
                    memory_access_bytes = (
                        ((hyperparams['n_centroids'] // hyperparams['vec_len']) * 2.03 * 1e9 * hyperparams['lut_bit'] * hyperparams['compress_lut_factor'] // 8) +
                        (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 3 + model_config['intermediate_size']) * 
                        hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8) // hyperparams['vec_len'])
                    )
                else:
                    memory_access_bytes = (
                        (np.log2(hyperparams['n_centroids']) * 2.03 * 1e9 / hyperparams['vec_len'] // 8) +
                        (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 4 + model_config['intermediate_size']) * 
                        hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8)) * 2 +
                        2.03 * 1e9)

                # Skip if no memory access (avoid division by zero)
                if memory_access_bytes == 0:
                    continue
                
                # Calculate operational intensity (OPS/Byte)
                operational_intensity = total_ops / memory_access_bytes
                
                # Calculate throughput (GFLOPS/s)
                total_data_load = memory_access_bytes
                off_chip_latency = total_data_load / fpga_config['off_chip_bw'] / (1024 ** 3)
                latency_sec = max(model_estimate[3] / 250 / 1e6, off_chip_latency)  # Convert cycles to seconds
                if latency_sec > 0:
                    throughput = (total_ops / 1e9) / latency_sec  # GFLOPS/s
                else:
                    continue
                
                operational_intensities.append(operational_intensity)
                throughputs.append(throughput)
                
                mode_str = "Decoding" if decoding else "Prefill"
                labels.append(f"SeqLen={seq_len}, {mode_str}")
                
            except Exception as e:
                print(f"Skipping seq_len={seq_len}, decoding={decoding}: {e}")
                continue
    
    return operational_intensities, throughputs, labels

def plot_roofline_model(operational_intensities, throughputs, labels, fpga_config, dsp_count):
    """
    Plot throughput vs operational intensity for different configurations.
    """
    # Create the plot
    plt.figure(figsize=(8, 4.3))
    
    # Plot actual performance points
    colors = plt.cm.tab10(np.linspace(0, 1, len(operational_intensities)))
    
    for i, (oi, tp, label) in enumerate(zip(operational_intensities, throughputs, labels)):
        if "Decoding" in label:
            marker = 'o'
            alpha = 0.8
        else:
            marker = 's'
            alpha = 0.6
        
        plt.scatter(oi, tp, c=[colors[i]], marker=marker, s=100, alpha=alpha, 
                   label=label)
    
    # Formatting
    plt.xlabel('Operational Intensity (OPS/Byte)', fontsize=12)
    plt.ylabel('Throughput (GFLOPS)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Use log scale for better visualization
    plt.xscale('log')
    plt.yscale('log')
    
    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Set reasonable axis limits
    if operational_intensities and throughputs:
        plt.xlim(min(operational_intensities) * 0.5, max(operational_intensities) * 2)
        plt.ylim(min(throughputs) * 0.5, max(throughputs) * 2)
    
    # Add computational and memory rooflines
    # Computational roof: number of DSPs * 2 ops per DSP * clock frequency
    # computational_peak = dsp_count * 2 * 250 * 1e6 / 1e9  # Convert to GFLOPS

    full_speed_peak = fpga_config['dsp'] * 2 * 250 * 1e6 / 1e9  # Convert to GFLOPS
    
    # Memory bandwidth roof: 820 GB/s
    memory_bandwidth = fpga_config['off_chip_bw']  # GB/s

    # Create x-axis range for roofline visualization
    x_range = np.linspace(0, 1000, 1000)
    
    # Plot the roofline model
    # plt.plot(x_range, [computational_peak] * len(x_range), 'g--', linewidth=2, label=f'Used Compute Bound: {computational_peak:.1f} GFLOPS')
    plt.plot(x_range, [full_speed_peak] * len(x_range), 'r--', linewidth=2, label=f'Full Speed Compute Bound: {full_speed_peak:.1f} GFLOPS')
    plt.plot(x_range, [oi * memory_bandwidth for oi in x_range], 'b--', linewidth=2, label=f'Memory Bound: {memory_bandwidth} GB/s')
    
    # Calculate and mark the ridge point (where memory bound meets compute bound)
    ridge_point = full_speed_peak / memory_bandwidth
    plt.scatter([ridge_point], [full_speed_peak], marker='*', color='black', s=150, zorder=10, 
                label=f'Ridge Point: {ridge_point:.2f} OPS/Byte')
    
    # Add annotations
    plt.annotate('Compute Bound', xy=(ridge_point*1.2, full_speed_peak*0.7), 
                 xytext=(ridge_point*1.2, full_speed_peak*0.7), fontsize=14, color='r')
    plt.annotate('Memory Bound', xy=(ridge_point*0.3, full_speed_peak*0.2), 
                 xytext=(ridge_point*0.3, full_speed_peak*0.2), fontsize=14, color='b')
    
    plt.tight_layout()
    plt.savefig('roofline_model.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    print("Performance plot saved as 'roofline_model.png'")

def compare_param_settings(model_config, fpga_config, param_dir='param_settings'):
    """
    Compare throughputs across different parameter settings for various sequence lengths.
    
    Args:
        model_config: Model configuration dictionary
        fpga_config: FPGA configuration dictionary
        param_dir: Directory containing parameter JSON files
    """
    
    # Get all JSON files in param_settings directory
    # param_files = glob.glob(os.path.join(param_dir, '*.json'))
    # # Sort parameter files alphabetically by filename
    # param_files.sort(key=lambda x: os.path.basename(x))
    # if not param_files:
    #     print(f"No JSON files found in {param_dir}")
    #     return

    param_files = ['param_settings/setting_1.json', 'param_settings/setting_w_vq.json']
    param_names = ['Act. VQ', 'Act. + Weight VQ']
    
    # Sequence lengths to test (powers of 2 from 16 to 8192)
    seq_lengths = [2**i for i in range(4, 14)]  # 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
    
    # Load tech config for DSP per float
    tech_config = load_config('fpga_resource_config.json')
    dsp_per_float = 1
    for dsp_conf in tech_config['dsp']:
        if dsp_conf['version'] == fpga_config['dsp_version']:
            dsp_per_float = dsp_conf['operations'][0]['count']
            break
    
    bram_bytes = fpga_config['bram'] * 2304
    uram_bytes = fpga_config['uram'] * 36864
    lutram_bytes = fpga_config['lut'] // 2 * 8
    max_memory_bytes = bram_bytes + uram_bytes + lutram_bytes
    
    # Store results for each parameter file and sequence length
    results = {}
    
    for param_file, param_name in zip(param_files, param_names):
        # param_name = os.path.basename(param_file).replace('.json', '')
        hyperparams = load_config(param_file)
        results[param_name] = {
            'prefill': [],
            'decoding': [],
            'seq_lengths': seq_lengths
        }
        
        for seq_len in seq_lengths:
            for decoding in [False, True]:
                try:
                    # Compute operations
                    _, total_ops = compute_model_operations(
                        seq_len=seq_len,
                        hidden_dim=model_config['hidden_size'],
                        intermediate_dim=model_config['intermediate_size'],
                        head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                        n_q_head=model_config['num_attention_heads'],
                        n_kv_head=model_config['num_key_value_heads'],
                        n_layer=model_config['num_hidden_layers'],
                        decoding=decoding
                    )
                    
                    # Compute model estimate
                    model_estimate, _ = QwenModel(
                        seq_len=seq_len,
                        hidden_dim=model_config['hidden_size'],
                        intermediate_dim=model_config['intermediate_size'],
                        head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                        n_q_head=model_config['num_attention_heads'],
                        n_kv_head=model_config['num_key_value_heads'],
                        n_layer=model_config['num_hidden_layers'],
                        vec_len=hyperparams['vec_len'],
                        n_centroids=hyperparams['n_centroids'],
                        pea_x=hyperparams['pea_x'],
                        pea_y=hyperparams['pea_y'],
                        pea_z=hyperparams['pea_z'],
                        max_lutl=fpga_config['lut']//2,
                        max_mem=max_memory_bytes,
                        max_bank=fpga_config['bram']+fpga_config['uram']+256,
                        parallel_acc=hyperparams['parallel_acc'],
                        off_chip_port=fpga_config['off_chip_port'],
                        dsp_per_float=dsp_per_float,
                        CCU_reload_factor=hyperparams['CCU_reload_factor'],
                        act_bit=hyperparams['act_bit'],
                        lut_bit=hyperparams['lut_bit'],
                        nonlinear_para_factor=hyperparams['nonlinear_para_factor'],
                        decoding=decoding,
                        weight_vq=hyperparams['weight_vq']
                    )
                    
                    # Check resources
                    _, spill_data_size = check_resources(model_estimate, fpga_config, verbose=False)
                    
                    # Calculate throughput
                    if "weight_vq" not in hyperparams or not hyperparams['weight_vq']:
                        total_data_load = (
                            ((hyperparams['n_centroids'] // hyperparams['vec_len']) * 2.03 * 1e9 * hyperparams['lut_bit'] * hyperparams['compress_lut_factor'] // 8) +
                            (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 3 + model_config['intermediate_size']) * 
                            hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8) // hyperparams['vec_len']) +
                            spill_data_size
                        )
                    else:
                        total_data_load = (
                            (np.log2(hyperparams['n_centroids']) * 2.03 * 1e9 / hyperparams['vec_len'] // 8) +
                            (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 3 + model_config['intermediate_size']) * 
                            hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8)) * 2 +
                            2.03 * 1e9) + spill_data_size
                    off_chip_latency = total_data_load / fpga_config['off_chip_bw'] / (1024 ** 3)
                    latency_sec = max(model_estimate[3] / 250 / 1e6, off_chip_latency)
                    throughput = (total_ops / 1e9) / latency_sec if latency_sec > 0 else 0
                    
                    if decoding:
                        results[param_name]['decoding'].append(throughput)
                    else:
                        results[param_name]['prefill'].append(throughput)
                        
                except Exception as e:
                    print(f"Error processing {param_name} with seq_len={seq_len}, decoding={decoding}: {e}")
                    if decoding:
                        results[param_name]['decoding'].append(0)
                    else:
                        results[param_name]['prefill'].append(0)
    
    #baseline
    results['FP16'] = {
        'prefill': [],
        'decoding': [],
        'seq_lengths': seq_lengths
    }
    for seq_len in seq_lengths:
        for decoding in [False, True]:
            # Compute operations
            _, total_ops = compute_model_operations(
                seq_len=seq_len,
                hidden_dim=model_config['hidden_size'],
                intermediate_dim=model_config['intermediate_size'],
                head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                n_q_head=model_config['num_attention_heads'],
                n_kv_head=model_config['num_key_value_heads'],
                n_layer=model_config['num_hidden_layers'],
                decoding=decoding
            )
            if not decoding:
                base_latency = total_ops / 5.83 / 1e12
            else:
                base_latency = total_ops / 5.83 / 1e12 * 6.5
            off_chip_latency = 2.03 * 1e9 * 2 / fpga_config['off_chip_bw'] / (1024 ** 3)
            latency_sec = max(base_latency, off_chip_latency)
            throughput = (total_ops / 1e9) / latency_sec if latency_sec > 0 else 0

            if decoding:
                results['FP16']['decoding'].append(throughput)
            else:
                results['FP16']['prefill'].append(throughput)
    
    #w4a8
    results['W4A8'] = {
        'prefill': [],
        'decoding': [],
        'seq_lengths': seq_lengths
    }
    for seq_len in seq_lengths:
        for decoding in [False, True]:
            # Compute operations
            ops_breakdown, total_ops = compute_model_operations(
                seq_len=seq_len,
                hidden_dim=model_config['hidden_size'],
                intermediate_dim=model_config['intermediate_size'],
                head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                n_q_head=model_config['num_attention_heads'],
                n_kv_head=model_config['num_key_value_heads'],
                n_layer=model_config['num_hidden_layers'],
                decoding=decoding
            )
            attention_ops = ops_breakdown['attention_qk'] + ops_breakdown['attention_av']
            base_latency = (total_ops-attention_ops) / 24.96 / 1e12 + (total_ops-attention_ops) / 2 / 5.83 / 1e12 + attention_ops / 5.83 / 1e12
            off_chip_latency = 2.03 * 1e9 / 2 / fpga_config['off_chip_bw'] / (1024 ** 3) * 3.5
            latency_sec = max(base_latency, off_chip_latency)
            throughput = (total_ops / 1e9) / latency_sec if latency_sec > 0 else 0

            if decoding:
                results['W4A8']['decoding'].append(throughput)
            else:
                results['W4A8']['prefill'].append(throughput)

    #weight vq only
    results['Weight VQ'] = {
        'prefill': [],
        'decoding': [],
        'seq_lengths': seq_lengths
    }
    for seq_len in seq_lengths:
        for decoding in [False, True]:
            # Compute operations
            ops_breakdown, total_ops = compute_model_operations(
                seq_len=seq_len,
                hidden_dim=model_config['hidden_size'],
                intermediate_dim=model_config['intermediate_size'],
                head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
                n_q_head=model_config['num_attention_heads'],
                n_kv_head=model_config['num_key_value_heads'],
                n_layer=model_config['num_hidden_layers'],
                decoding=decoding
            )
            attention_ops = ops_breakdown['attention_qk'] + ops_breakdown['attention_av']
            if not decoding:
                base_latency = total_ops / 5.6 / 1e12
            else:
                base_latency = (total_ops - attention_ops) / 5.6 / 1e12 * 6.5 + attention_ops / 5.6 / 1e12
            off_chip_latency = (2.03 * 1e9 / 4 + 2.03 * 1e9 / 16) / fpga_config['off_chip_bw'] / (1024 ** 3)
            latency_sec = max(base_latency, off_chip_latency)
            throughput = (total_ops / 1e9) / latency_sec if latency_sec > 0 else 0

            if decoding:
                results['Weight VQ']['decoding'].append(throughput)
            else:
                results['Weight VQ']['prefill'].append(throughput)

    # Create separate plots for prefill and decoding
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    n_seq_lengths = len(seq_lengths)
    
    # Create prefill plot
    fig, axes = plt.subplots(2, (n_seq_lengths + 1) // 2, figsize=(20, 10))
    if n_seq_lengths == 1:
        axes = [axes]
    elif (n_seq_lengths + 1) // 2 == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        
        # Plot prefill bars
        for i, (param_name, data) in enumerate(results.items()):
            seq_idx = seq_lengths.index(seq_len)
            if seq_idx < len(data['prefill']):
                ax.bar(i, data['prefill'][seq_idx], color=colors[i], alpha=0.7, 
                        width=0.8, label=param_name)
        
        ax.set_title(f'Sequence Length: {seq_len}', fontsize=12)
        ax.set_ylabel('Throughput (GFLOPS)', fontsize=10)
        ax.set_xlabel('Parameter Settings', fontsize=10)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([name for name in results.keys()], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Remove empty subplots if any
    for idx in range(n_seq_lengths, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Prefill Throughput Comparison Across Parameter Settings', fontsize=16)
    plt.tight_layout()
    plt.savefig('param_comparison_prefill.png', dpi=500)
    plt.close()
    
    print("Prefill parameter comparison plot saved as 'param_comparison_prefill.png'")
    
    # Create decoding plot
    fig, axes = plt.subplots(2, (n_seq_lengths + 1) // 2, figsize=(20, 10))
    if n_seq_lengths == 1:
        axes = [axes]
    elif (n_seq_lengths + 1) // 2 == 1:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    for idx, seq_len in enumerate(seq_lengths):
        ax = axes[idx]
        
        # Plot decoding bars
        for i, (param_name, data) in enumerate(results.items()):
            seq_idx = seq_lengths.index(seq_len)
            if seq_idx < len(data['decoding']):
                ax.bar(i, data['decoding'][seq_idx], color=colors[i], alpha=0.7,
                        width=0.8, label=param_name)
        
        ax.set_title(f'Sequence Length: {seq_len}', fontsize=12)
        ax.set_ylabel('Throughput (GFLOPS)', fontsize=10)
        ax.set_xlabel('Parameter Settings', fontsize=10)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels([name for name in results.keys()], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Remove empty subplots if any
    for idx in range(n_seq_lengths, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Decoding Throughput Comparison Across Parameter Settings', fontsize=16)
    plt.tight_layout()
    plt.savefig('param_comparison_decoding.png', dpi=500)
    plt.close()
    
    print("Decoding parameter comparison plot saved as 'param_comparison_decoding.png'")
    
    # Also create a line plot showing throughput vs sequence length
    plt.figure(figsize=(8.5, 4.5))
    
    base = results['Act. VQ']['decoding'][0]
    for i, (param_name, data) in enumerate(results.items()):
        data['prefill'] = np.array(data['prefill']) / base
        data['decoding'] = np.array(data['decoding']) / base
        plt.plot(seq_lengths[:len(data['prefill'])], data['prefill'], 
                marker='o', color=colors[i], linewidth=2, markersize=6,
                label=f'{param_name} (Prefill)')
        plt.plot(seq_lengths[:len(data['decoding'])], data['decoding'], 
                marker='s', color=colors[i], linewidth=2, markersize=6,
                linestyle='--', alpha=0.7,
                label=f'{param_name} (Decode)')
    
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Normalized Throughput', fontsize=12)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(seq_lengths, [str(sl) for sl in seq_lengths])
    
    plt.tight_layout()
    plt.savefig('throughput_vs_seqlen_final.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    print("Throughput vs sequence length plot saved as 'throughput_vs_seqlen.png'")

def main():
    parser = argparse.ArgumentParser(description='Performance Modeling and Resource Estimation')
    parser.add_argument('--model_config', type=str, default='qwen2.5_0.5b.json', help='Path to the language model configuration JSON')
    parser.add_argument('--fpga_config', type=str, default='v80.json', help='Path to the FPGA hardware configuration JSON')
    parser.add_argument('--hyperparams', type=str, default='inference_param.json', help='Path to the hyperparameters JSON')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length (up to the next decoding token)')
    parser.add_argument('--decoding', action='store_true', default=False, help='Modeling for decoding phase')
    parser.add_argument('--roofline', action='store_true', default=False, help='Generate roofline model plot')
    parser.add_argument('--compare_params', action='store_true', default=False, help='Compare different parameter settings')
    args = parser.parse_args()

    # Load configurations
    model_config = load_config(args.model_config)
    fpga_config = load_config(args.fpga_config)
    hyperparams = load_config(args.hyperparams)

    tech_config = load_config('fpga_resource_config.json')
    dsp_per_float = 1
    for dsp_conf in tech_config['dsp']:
        if dsp_conf['version'] == fpga_config['dsp_version']:
            dsp_per_float = dsp_conf['operations'][0]['count']
            break
    
    bram_bytes = fpga_config['bram'] * 2304
    uram_bytes = fpga_config['uram'] * 36864
    lutram_bytes = fpga_config['lut'] // 2 * 8
    max_memory_bytes = bram_bytes + uram_bytes + lutram_bytes

    model_estimate, breakdown = QwenModel(
        seq_len=args.seq_len,
        hidden_dim=model_config['hidden_size'], # hidden dimension
        intermediate_dim=model_config['intermediate_size'], # dimension of FFN layer
        head_dim=model_config['hidden_size'] // model_config['num_attention_heads'], # head dimension
        n_q_head=model_config['num_attention_heads'], # number of heads in total
        n_kv_head=model_config['num_key_value_heads'], # number of KV heads
        n_layer=model_config['num_hidden_layers'], # number of layers
        vec_len=hyperparams['vec_len'], # LUTLinear: vector length
        n_centroids=hyperparams['n_centroids'], # LUTLinear: number of centroids per LUT
        pea_x=hyperparams['pea_x'], # GEMM: number of PE in x direction
        pea_y=hyperparams['pea_y'], # GEMM: number of PE in y direction
        pea_z=hyperparams['pea_z'], # GEMM: number of PE in z direction
        max_lutl=fpga_config['lut']//2, # device: max LUT in SLICEL CLB
        max_mem=max_memory_bytes, # device: total PL memory
        max_bank=fpga_config['bram']+fpga_config['uram']+256, # device: number of memory banks (LUTRAM count 1 bank per LUT)
        parallel_acc=hyperparams['parallel_acc'], # LUTLinear: number of parallel accumulations
        off_chip_port=fpga_config['off_chip_port'], # device: off-chip port count
        dsp_per_float=dsp_per_float, # DSP per float
        CCU_reload_factor=hyperparams['CCU_reload_factor'], # CCU reduce factor
        act_bit=hyperparams['act_bit'], # LUTLinear: bitwidth of activation
        lut_bit=hyperparams['lut_bit'], # LUTLinear: bitwidth of LUT
        nonlinear_para_factor=hyperparams['nonlinear_para_factor'], # factor for nonlinear operations
        decoding=args.decoding, # whether to decode
        weight_vq=hyperparams['weight_vq'] # whether to use weight VQ
    )

    print(f"dsp: {model_estimate[0]}, lut: {model_estimate[1]}, memory: {model_estimate[2] + model_estimate[4] + model_estimate[5]} bytes, latency: {model_estimate[3]} cycles")
    
    

    # Print breakdown with pretty formatting
    print("\n" + "="*50)
    print("LATENCY BREAKDOWN")
    print("="*50)
    
    total_breakdown_latency = sum(breakdown.values())-breakdown['gqa_qk']
    
    for component, latency in breakdown.items():
        percentage = (latency / total_breakdown_latency * 100) if total_breakdown_latency > 0 else 0
        print(f"{component.upper():15} : {latency:12,.0f} cycles ({percentage:5.1f}%)")
    
    print("-"*50)
    print(f"{'TOTAL':15} : {total_breakdown_latency:12,.0f} cycles (100.0%)")
    print("="*50)

    # Compute total operations
    ops_breakdown, total_ops = compute_model_operations(
        seq_len=args.seq_len,
        hidden_dim=model_config['hidden_size'],
        intermediate_dim=model_config['intermediate_size'],
        head_dim=model_config['hidden_size'] // model_config['num_attention_heads'],
        n_q_head=model_config['num_attention_heads'],
        n_kv_head=model_config['num_key_value_heads'],
        n_layer=model_config['num_hidden_layers'],
        decoding=args.decoding
    )

    # Print total FLOPs
    total_gflops = total_ops / 1e9
    print(f"Total FLOPs: {total_gflops:.2f} GFLOPs")

    # Check resource constraints
    _, spill_data_size = check_resources(model_estimate, fpga_config)

    if "weight_vq" not in hyperparams or not hyperparams['weight_vq']:
        total_data_load = (
            ((hyperparams['n_centroids'] // hyperparams['vec_len']) * 2.03 * 1e9 * hyperparams['lut_bit'] * hyperparams['compress_lut_factor'] // 8) +
            (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 4 + model_config['intermediate_size']) * 
            hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8) // hyperparams['vec_len']) +
            spill_data_size
        )
    else:
        total_data_load = (
            (np.log2(hyperparams['n_centroids']) * 2.03 * 1e9 / hyperparams['vec_len'] // 8) +
            (model_config['num_hidden_layers'] * (model_config['hidden_size'] * 4 + model_config['intermediate_size']) * 
            hyperparams['n_centroids'] * (hyperparams['act_bit'] // 8)) * 2 +
            2.03 * 1e9) + spill_data_size
    off_chip_latency = total_data_load / fpga_config['off_chip_bw'] / (1024 ** 3) * 1000
    print(f"off-chip latency: {off_chip_latency} ms")
    print(f"on-chip latency: {model_estimate[3] / 250 / 1000} ms")
    print(f"inference latency: {max(model_estimate[3] / 250 / 1000, off_chip_latency)} ms") # 250MHz clock, assume fully overlap
    print(f"Throughput: {total_gflops * 1000 / max(model_estimate[3] / 250 / 1000, off_chip_latency)} GFLOPS/s")

    # Generate roofline model if requested
    if args.roofline:
        print("\nGenerating roofline model...")
        operational_intensities, throughputs, labels = generate_roofline_model(model_config, fpga_config, hyperparams)
        plot_roofline_model(operational_intensities, throughputs, labels, fpga_config, model_estimate[0])
    
    if args.compare_params:
        print("\nComparing parameter settings...")
        compare_param_settings(model_config, fpga_config)

if __name__ == "__main__":
    main()
