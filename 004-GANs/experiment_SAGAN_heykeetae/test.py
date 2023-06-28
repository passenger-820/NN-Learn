import os

# # 400_fake1.png
# # 400_fake2.png
# # 400_fake64.png
# # 800_fake1.png
# # ...
# # ...
# # 20000_fake64.png
# path = "./samples/sagan_8"
# files = os.listdir(path)
# files.sort(key=lambda x: (int( x.split('_')[0] ),  # 依据step(400~20000: +400)升序
#                           int( x.split('_fake')[1].split('.')[0] ) # 再依据i(1~64: +1) 升序
#                           ) )
# for filename in files:
#     print(filename)
