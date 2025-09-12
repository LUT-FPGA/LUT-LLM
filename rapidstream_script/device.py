from rapidstream import DeviceFactory

df = DeviceFactory(row=3, col=1, part_num="xcv80-lsva4737-2MHP-e-S")

for x in range(1):
    for y in range(3):
        if y == 0:
            pblock = f"-add CLOCKREGION_X{0}Y{0}:CLOCKREGION_X{9}Y{4}"
        else:
            pblock = f"-add CLOCKREGION_X{0}Y{(y-1)*3+5}:CLOCKREGION_X{9}Y{(y-1)*3+7}"

        df.set_slot_pblock(x, y, [pblock])

df.extract_slot_resources()

for x in range(1):
    df.set_slot_capacity(x, 0, north=20000)
    df.set_slot_capacity(x, 1, north=20000)

    df.set_slot_capacity(x, 1, south=20000)
    df.set_slot_capacity(x, 2, south=20000)

df.generate_virtual_device("v80_device.json")