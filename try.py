import matplotlib.pyplot as plt
from matplotlib import animation

fig, ax = plt.subplots()

# 定义存储数据的列表
xdata = []
ydata = []

# 接收line2D对象
line, = plt.plot(xdata, ydata, 'ro')


# 定义更新函数
def update(frame_ID):   # 帧
    """ 根据每一帧的ID： frame_ID来更新数据。这里由于是要一直连续画，因此需要apeend一下之后的数据"""

    # print(frame_ID)

    xdata.append(frame_ID)
    ydata.append(frame_ID ** 2)

    line.set_data(xdata, ydata)

    return line,


def init_figure():
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)


# 调用生成动画的函数生成动图
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=3,
    init_func=init_figure,
    interval=1000   #  每隔多少时间生成一帧图像，单位是ms
)

plt.show()   # 如果要保存视频和gif就不要show()