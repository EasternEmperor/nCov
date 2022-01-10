import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import ConnectionPatch
from pyecharts import options as opts
from pyecharts.charts import Map
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score


"""
读取文件内容："nCov_1.csv"-"nCov_15.csv"
"""
dataDic = {}            # 字典结构保存每一天的数据
for i in range(1, 16):
    df = pd.read_csv('nCov_{}.csv'.format(i), encoding = 'gbk')
    dataDic[i] = df
    
plt.rcParams['font.sans-serif'] = ['SimHei']    # 添加对中文字体的支持



""" 公用函数 """

""" 柱形图 """
def drawBars(rank, quantity):
    """ 绘制不同颜色的柱形图 """
    color = ['grey', 'gold', 'darkviolet', 'turquoise', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 
             'darkorange', 'lightgreen', 'plum', 'tan', 'khaki', 'pink', 'skyblue', 
             'lawngreen', 'salmon']
    for i in range(len(rank)):
        x, y = [], []
        x.append(rank[i])
        y.append(quantity[i])
        plt.bar(x, y, 0.8, color = color[i])

    plt.grid(True, linestyle = '--', alpha = 0.8)           # 设置网格线
    ax.spines['right'].set_visible(False)                   # 隐藏右边框
    ax.spines['top'].set_visible(False)                     # 隐藏上边框

""" 线图 """
def set_annotation():
    po_annotation = []
    for i in range(15):
        """ 为每个点设置注释 """
        x = i + 1           # 横坐标为日期
        y = worldSum[i]     # 纵坐标为世界总病例
        print(x, y)
        point, = plt.plot(x, y)
        # 设置单个点注释
        annotation = plt.annotate(('日期：12月{}日'.format(x), 'totalCases = ' + str(y)), xy = (x + 0.1, y + 0.1),
                                  xycoords = 'data', xytext = (x - 2.3, y + 10 ** 6), textcoords = 'data',
                                  horizontalalignment = 'left',
                                  arrowprops = dict(arrowstyle = 'simple', connectionstyle = 'arc3, rad = -0.1'),
                                  bbox = dict(boxstyle = 'round', facecolor = 'w', edgecolor = '0.5', alpha = 0.9))
        annotation.set_visible(False)
        po_annotation.append([point, annotation])
    return po_annotation

def on_move(event):
    """ 设置事件，当鼠标移动到点上时，显示po_annotation的内容 """
    visibility_changed = False
    for point, annotation in po_annotation:
        should_be_visible = (point.contains(event)[0] == True)
        if should_be_visible != annotation.get_visible():
            visibility_changed = True
            annotation.set_visible(should_be_visible)
    if visibility_changed:
        plt.draw()

""" 地图 """
def map_visualmap(countries, cases) -> map:
    """ 绘制世界地图 """
    c = (
        Map()
        .add("各国新冠确诊数", [list(z) for z in zip(countries, cases)], "world")
        .set_global_opts(
            title_opts = opts.TitleOpts(title = "截止2020.12.15世界新冠疫情分布情况"),
            visualmap_opts = opts.VisualMapOpts(min_ = min(cases[0 : 100]), max_ = max(cases) + 10 ** 5))
        .set_series_opts(label_opts = opts.LabelOpts(is_show = False))
        )
    return c

""" 应对国家求分数 """
def get_score(last):
    countries = last['country'].tolist()
    for i in range(len(last)):
        """ 按排名给分，排名越靠后分值越高 """
        score[countries[i]] += i



""" 1、15天中，新冠疫情的总体变化趋势：使用线图 """
print('--------------------------------------------1--------------------------------------------\n\n')

worldSum = []           # 保存每天世界病例总和
for i in range(15):
    """ 对每天世界总病例求和 """
    worldSum.append(dataDic[i + 1]['totalCases'].sum())

fig = plt.figure(figsize = (10, 10))            # 窗口大小
fig.canvas.set_window_title('Global Trends')    # 窗口标题
ax = plt.axes()                                 # 创建坐标对象
ax.set_xlabel('日期')
ax.set_ylabel('TotalCases')
ax.set_title('2020年12.1-12.15世界新冠疫情变化情况')   # 线图标题

""" 设置x坐标标签 """
days = [i for i in range(1, 16)]        # 日期
ax.set_xticks(days)
xlabel = ['2020.12.{}'.format(i) for i in range(1, 16)]
ax.set_xticklabels(xlabel, rotation = 300, fontsize = 8)

# 绘制线图：颜色为红色、宽度为1像素的连续直线，数据点为正方点
plt.plot(days, worldSum, 's', color = 'red', linewidth = 2.0, linestyle = '-', label = 'line', clip_on = False)

# 绘制趋势线：颜色为蓝色
z = np.polyfit(days, worldSum, 2)       # 拟合为二次函数
p = np.poly1d(z)
plt.plot(days, p(days), 'b--', alpha = 0.5)
equation = "y = {}x^2 + {}x + {}".format(round(z[0], 2), round(z[1], 2), round(z[2], 2))    # 二次方程公式


""" 设置鼠标移动到点上时，显示点坐标（日期，PM） """
po_annotation = set_annotation()

plt.xlim(1, 15.1)                                       # 设置横轴的日期
ax.spines['right'].set_visible(False)                   # 隐藏右边框
ax.spines['top'].set_visible(False)                     # 隐藏上边框
plt.legend(['The World Trends', '拟合曲线：{}'.format(equation)], loc = 'upper left')    # 设置图例
plt.grid(True, linestyle = '--', alpha = 0.8)           # 设置网格线

on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)




""" 2、累计确诊数排名前20的国家名称及其数量：使用柱形图 """
print('\n\n--------------------------------------------2--------------------------------------------\n\n')



countries = dataDic[15].copy(deep = True)     # 取第15天的数据进行比较，深复制

# 将各国数据按确诊总数降序排序，且空数据国家排在最后
countries.sort_values('totalCases', ascending = False, inplace = True, na_position = 'last')
print(countries)

fig = plt.figure(figsize = (10, 10))                        # 窗口大小
fig.canvas.set_window_title('Top 20 Countries')             # 窗口标题
ax = plt.axes()                                             # 创建坐标对象
ax.set_title('截止2020.12.15新冠确诊人数排名前20的国家')    # 线图标题
ax.set_xlabel('Ranking', family = 'Times New Roman')
ax.set_ylabel('TotalCases', family = 'Times New Roman')

""" 设置x轴坐标标签 """
rank = [i for i in range(1, 21)]                            # 排名
top20 = countries['totalCases'].tolist()[0 : 20]            # 前20的国家的确诊数
ax.set_xticks(rank)
xlabel = countries['country'].tolist()[0 : 20]              # 国家名称
ax.set_xticklabels(xlabel)
plt.xticks(rotation = 300)                                  # 使x坐标标签斜着，以防重叠
print(top20)
print(xlabel)

""" 绘制20条不同颜色的柱状 """

drawBars(rank, top20)
for a, b in zip(rank, top20):
    """设置排名标签"""
    plt.text(a, b + 10 ** 5, top20[a - 1], family = 'Times New Roman',
             ha = 'center', va = 'bottom', fontsize = 10, rotation = -300)
plt.legend(xlabel, loc = 'upper right')




""" 3、15 天中，每日新增确诊数累计排名前 10 个国家的每日新增确诊数据的曲线图 """
print('\n\n--------------------------------------------3--------------------------------------------\n\n')


""" 15天内每日新增确诊累计 = 第15天累计确诊 - 第1天累计确诊 """
first = dataDic[1].copy(deep = True)      # 第1天数据
last = dataDic[15].copy(deep = True)      # 第15天数据
last['CumulDiag'] = last['totalCases'] - first['totalCases']    # 新增列"CumulDiag"：15日内新增确诊累计

# 将last中的数据按CumulDiag降序排序，空数据排在最后位
last.sort_values('CumulDiag', ascending = False, inplace = True, na_position = 'last')
print(last)

# 设置画图属性
fig = plt.figure(figsize = (10, 10))            # 窗口大小
fig.canvas.set_window_title('Top 10 Countries') # 窗口标题

""" 设置x轴标签 """
days = [i for i in range(1, 16)]                # 1-15
countries = last['country'].tolist()[0 : 10]    # countries储存排名前10的国家名称
print(countries)

order = last['country'].tolist()                # 排序标准
top10 = []                                      # top10储存排名前10国家每日新增确诊数
for j in range(10):
    """ 计算得到每日该10个国家的新增病例 """
    newcases = []
    for i in range(1, 15):
        """ 由于只爬取了1-15日的数据，只能计算出1-14日的新增 """
        preDay = dataDic[i].copy(deep = True)                 # 前一天的数据，深复制
        latterDay = dataDic[i + 1].copy(deep = True)          # 后一天的数据，深复制

        # 先将其中的数据按照order的顺序排序
        preDay['country'] = preDay['country'].astype('category')            # 设为"category"数据类型以便按照last中的country顺序排序
        latterDay['country'] = latterDay['country'].astype('category')      # 同上
        preDay['country'].cat.reorder_categories(order, inplace = True)     # 按照order中的顺序对preDay的"country"列进行重新排序
        latterDay['country'].cat.reorder_categories(order, inplace = True)  # 同上
        preDay.sort_values('country', inplace = True)
        latterDay.sort_values('country', inplace = True)

        # 求出第i天第j个国家的新增病例
        latter = latterDay.iloc[j, 1]         # 获得第j个国家的totalCasees
        pre = preDay.iloc[j, 1]
        newcases.append(latter - pre)

    # 15日的新增病例用爬取到的数据
    newcases.append(last['newCases'].tolist()[j])
    top10.append(newcases)
print(top10)

""" 绘制线图 """

""" 第一张图：将十个国家放在一起 """

ax = plt.axes()     # 创建坐标对象
ax.set_title('2020.12.1-12.15新冠新增确诊排名前10的国家每日新增确诊曲线')    # 线图标题
ax.set_xlabel('Date', family = 'Times New Roman')
ax.set_ylabel('NewCases', family = 'Times New Roman')
# 设置x坐标标签
ax.set_xticks(days)
xlabel = ['2020.12.{}'.format(i) for i in range(1, 16)]
ax.set_xticklabels(xlabel, rotation = 300, fontsize = 8)

color = ['grey', 'gold', 'darkviolet', 'turquoise', 'r', 'g', 'b', 'c', 'm', 'y']   # 每个国家的颜色
makers = ['o', '^', '2', 'p', '+', 'D', '-', 's', '*', 'x']                         # 点的形状

for i in range(10):
    plt.plot(days, top10[i], makers[i], color = color[i], linewidth = 2.0, linestyle = '-', 
             label = 'line{}'.format(i), clip_on = False)


""" 设置鼠标移动到点上时，显示点坐标（国家，日期，增长数） """
po_annotation = []
for i in range(10):
    for j in range(1, 16):
        x = j                   # 横坐标为日期
        y = top10[i][j - 1]     # 纵坐标为增长数
        point, = plt.plot(x, y)
        annotation = plt.annotate(('国家：' + countries[i], '时间：2020.12.{}'.format(j), 'NewCases = {}'.format(y)),
                                  xy = (x + 0.1, y + 0.1), xycoords = 'data', xytext = (x - 3.0, y - 2 * (10 ** 4)),
                                  textcoords = 'data', ha = 'left',
                                  arrowprops = dict(arrowstyle = 'simple', connectionstyle = 'arc3, rad = -0.1'),
                                  bbox = dict(boxstyle = 'round', facecolor = 'w', edgecolor = '0.5', alpha = 0.9))
        annotation.set_visible(False)
        po_annotation.append([point, annotation])

plt.xlim(1, 15.1)                                       # 设置横轴的日期
ax.spines['right'].set_visible(False)                   # 隐藏右边框
ax.spines['top'].set_visible(False)                     # 隐藏上边框
plt.grid(True, linestyle = '--', alpha = 0.8)           # 设置网格线
# 设置图例，放在图外
plt.legend(countries, loc = 'upper left', bbox_to_anchor = (1.005, 1.12), borderaxespad = 0.)

on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)


""" 第二张图：分两个子图，一张图只有美国，另一张图绘制剩余9个国家 """

# 设置画图属性
fig = plt.figure(figsize = (10, 10))            # 窗口大小
fig.canvas.set_window_title('Top 10 Countries') # 窗口标题
fig.subplots_adjust(wspace = 0)                 # 调整子图空间

# 第一个子图：只有美国
plt.subplot(121)
ax = plt.gca()          # 创建坐标对象
ax.set_title('2020.12.1-12.15新冠新增确诊排名1st的美国每日新增确诊曲线')    # 线图标题
ax.set_xlabel('Date', family = 'Times New Roman')
ax.set_ylabel('NewCases', family = 'Times New Roman')
# 设置x坐标标签
ax.set_xticks(days)
xlabel = ['2020.12.{}'.format(i) for i in range(1, 16)]
ax.set_xticklabels(xlabel, rotation = 300, fontsize = 8)
plt.plot(days, top10[0], makers[0], color = color[0], linewidth = 2.0, linestyle = '-',
         label = 'line{}'.format(0), clip_on = False)

plt.grid(True, linestyle = '--', alpha = 0.8)   # 设置网格线
# 设置图例，放在图外
plt.legend(countries[0], loc = 'upper left', bbox_to_anchor = (0, 1.0), borderaxespad = 0.)

# 第二个子图：剩余九个国家
plt.subplot(122)
ax = plt.gca()          # 创建坐标对象
ax.set_title('2020.12.1-12.15新冠新增确诊排名2-9的国家每日新增确诊曲线')    # 线图标题
ax.set_xlabel('Date', family = 'Times New Roman')
ax.set_ylabel('NewCases', family = 'Times New Roman')
# 设置x坐标标签
ax.set_xticks(days)
xlabel = ['2020.12.{}'.format(i) for i in range(1, 16)]
ax.set_xticklabels(xlabel, rotation = 300, fontsize = 8)
for i in range(1, 10):
    plt.plot(days, top10[i], makers[i], color = color[i], linewidth = 2.0, linestyle = '-', 
             label = 'line{}'.format(i), clip_on = False)
    
plt.grid(True, linestyle = '--', alpha = 0.8)   # 设置网格线
# 设置图例，放在图外
plt.legend(countries[1 : 10], loc = 'upper left', bbox_to_anchor = (1.025, 1.12), borderaxespad = 0.)




""" 4、累计确诊人数占国家总人口比例最高的10个国家：柱状图 """
print('\n\n--------------------------------------------4--------------------------------------------\n\n')


""" 累计确诊：直接看第15天的数据 """
last = dataDic[15].copy(deep = True)
last['Case/Pop'] = last['totalCases'] / last['population']      # 新增列：为累计确诊占总人口比

# 将last中的数据按Case/Pop降序排序，空数据排在最后位
last.sort_values('Case/Pop', ascending = False, inplace = True, na_position = 'last')
print(last)

# 设置画图属性
fig = plt.figure(figsize = (10, 10))                            # 窗口大小
fig.canvas.set_window_title('Proportion Top 10 Countries')      # 窗口标题
ax = plt.axes()                                                 # 创建坐标对象
plt.title('截止2020.12.15累计确诊人数占国家总人口比例最高的10个国家')         # 柱状图标题
ax.set_xlabel('Ranking', family = 'Times New Roman')
ax.set_ylabel('Case / Pop', family = 'Times New Roman')


""" 绘制柱状图：展示比例 """

""" 设置x轴坐标标签 """
rank = [i for i in range(1, 11)]
proportion = last['Case/Pop'].tolist()[0 : 10]                  # 10个国家的比例
countries = last['country'].tolist()[0 : 10]                    # countries储存排名前10的国家名称
ax.set_xticks(rank)
ax.set_xticklabels(countries)
plt.xticks(rotation = 300)                                      # 使x坐标标签斜着，以防重叠
print(countries)
print(proportion)

""" 绘制10条不同颜色的柱状 """

drawBars(rank, proportion)
for a, b in zip(rank, proportion):
    """设置排名标签"""
    plt.text(a, b + 0.001, '{}%'.format(round(proportion[a - 1] * 100, 2)), family = 'Times New Roman',
             ha = 'center', va = 'bottom', fontsize = 10, rotation = -300)
plt.legend(countries, loc = 'upper right')




""" 5、死亡率（累计死亡人数/累计确诊人数）最低的10个国家：柱状图 """
print('\n\n--------------------------------------------5--------------------------------------------\n\n')


""" 死亡率：以第15天最终结果计 """
last = dataDic[15].copy(deep = True)
last['deathRate'] = last['totalDeaths'] / last['totalCases']        # 新增一列：为死亡率

# 将last中的数据按deathRate升序排序，空数据排在最后位
last.sort_values('deathRate', ascending = True, inplace = True, na_position = 'last')
print(last)

# 设置画图属性
fig = plt.figure(figsize = (10, 10))                            # 窗口大小
fig.canvas.set_window_title('Death Rate Top 10 Countries')      # 窗口标题
ax = plt.axes()                                                 # 创建坐标对象
plt.title('截止2020.12.15死亡率最低的10个国家')                 # 柱状图标题
ax.set_xlabel('Ranking', family = 'Times New Roman')
ax.set_ylabel('DeathRate', family = 'Times New Roman')


""" 绘制柱状图：展示死亡率 """

""" 设置x轴坐标标签 """
rank = [i for i in range(1, 11)]
deathRate = last['deathRate'].tolist()[0 : 10]                  # 10个国家的比例
countries = last['country'].tolist()[0 : 10]                    # countries储存排名前10的国家名称
ax.set_xticks(rank)
ax.set_xticklabels(countries)
plt.xticks(rotation = 300)                                      # 使x坐标标签斜着，以防重叠
print(countries)
print(deathRate)

""" 绘制10条不同颜色的柱状 """

drawBars(rank, deathRate)
for a, b in zip(rank, deathRate):
    """设置排名标签"""
    plt.text(a, b + 0.0001, '{}%'.format(round(deathRate[a - 1] * 100, 2)), family = 'Times New Roman',
             ha = 'center', va = 'bottom', fontsize = 10, rotation = -300)
ax.legend(countries)




""" 6、用饼图展示各个国家的累计确诊人数的比例（你爬取的所有国家，数据较小的国家可以合并处理） """
print('\n\n--------------------------------------------6--------------------------------------------\n\n')


""" 处理最后一天数据，使之按累计确诊人数降序排序 """
allCases = dataDic[15]['totalCases'].sum()                      # 第15天世界确诊人数和
last = dataDic[15].copy(deep = True)
last.sort_values('totalCases', ascending = False, inplace = True, na_position = 'last')
print(last)


# 设置画图属性
fig = plt.figure(figsize = (10, 10), facecolor = 'lightgreen')  # 窗口大小和窗口底色
fig.canvas.set_window_title('Total Cases Rate Whole World')     # 窗口标题
ax1 = fig.add_subplot(121)                                      # 创建坐标对象
ax2 = fig.add_subplot(122)
fig.subplots_adjust(wspace = 0)                                 # 调整子图空间


""" 大饼图：包括排名前10个国家的数据+其他 """

quantity_1 = last['totalCases'].tolist()[0 : 10]                # 数据
quantity_1.append(allCases - sum(quantity_1))
labels_1 = last['country'].tolist()[0 : 10]                     # 标签
labels_1.append('其他')
explode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03]                  # 分裂距离
print(quantity_1, labels_1)

# 绘制饼图
ax1.pie(x = quantity_1, colors = ['grey', 'gold', 'darkviolet', 'turquoise', 'r', 'darkorange', 'plum', 'c', 'm', 'y', 'k'],
        explode = explode, autopct = '%1.2f%%', startangle = 60, labels = labels_1,
        textprops = {'color': 'b', 'fontsize': 10, 'rotation': -400,}, )


""" 小饼图：'其他'国家的（剩余所有国家）中的比例（6+其他） """

labels_2 = last['country'].tolist()[10 : 16]                    # 标签
labels_2.append('其他')
quantity_2 = last['totalCases'].tolist()[10 : 16]               # 数据
quantity_2.append(allCases - sum(quantity_2) - sum(quantity_1[0 : 10]))
print(quantity_2, labels_2)

# 绘制饼图
ax2.pie(x = quantity_2, colors = ['plum', 'tan', 'khaki', 'pink', 'skyblue', 'lawngreen', 'salmon'],
        autopct = '%1.2f%%', startangle = 60, labels = labels_2, radius = 0.5, shadow = True,
        textprops = {'color': 'b', 'fontsize': 10, 'rotation': -400,}, )


""" 用 ConnectionPath 画出两张饼图的间连线 """

# 饼图边缘的数据
theta1 = ax1.patches[-1].theta1
theta2 = ax1.patches[-1].theta2
center = ax1.patches[-1].center
r = ax1.patches[-1].r
width = 0.2

# 上边缘连线
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = np.sin(np.pi / 180 * theta2) + center[1]
con_a = ConnectionPatch(xyA = (-width / 2, 0.5), xyB = (x, y),
                        coordsA = 'data', coordsB = 'data',
                        axesA = ax2, axesB = ax1)

# 下边缘连线
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = np.sin(np.pi / 180 * theta1) + center[1]
con_b = ConnectionPatch(xyA = (-width / 2, -0.5), xyB = (x, y),
                        coordsA = 'data', coordsB = 'data',
                        axesA = ax2, axesB = ax1)

for con in [con_a, con_b]:
    """ 添加连线 """
    con.set_linewidth(1)            # 连线宽度
    con.set_color = ([0, 0, 0])     # 连线颜色
    ax2.add_artist(con)             # 连线

plt.suptitle('世界各国截止2020.12.15累计确诊人数占世界总确诊人数比例')
ax1.legend(loc = 'upper left', bbox_to_anchor = (-0.3, 1.12), borderaxespad = 0.)
ax2.legend()


    

""" 7、展示全球各个国家累计确诊人数的箱型图，要有平均值 """
print('\n\n--------------------------------------------7--------------------------------------------\n\n')


""" 累计确诊人数以第15天计 """
last = dataDic[15].copy(deep = True)
last.dropna(axis = 0, how = 'all', subset = ['totalCases'], inplace = True)
print(last['totalCases'])

""" 绘制箱型图 """


# 设置画图属性
fig = plt.figure(figsize = (10, 10))                            # 窗口大小和窗口底色
fig.canvas.set_window_title('Total Cases World Box Diagram')    # 窗口标题
plt.title('世界各国截止2020.12.15累计确诊人数箱型图')

ax = last.boxplot(column = ['totalCases'], meanline = True, showmeans = True, vert = False)
ax.text(last['totalCases'].mean(), 1.09, '平均数：{}'.format(last['totalCases'].mean()))
ax.text(last['totalCases'].median(), 1.11,  '中位数：{}'.format(last['totalCases'].median()))
ax.text(last['totalCases'].quantile(0.25), 0.88, 'Q1：{}'.format(last['totalCases'].quantile(0.25)))
ax.text(last['totalCases'].quantile(0.75), 0.9, 'Q3：{}'.format(last['totalCases'].quantile(0.75)))




""" 8、其余我想展示的数据 """
print('\n\n--------------------------------------------8--------------------------------------------\n\n')


""" a、累计检测量占国家总人口比例排名前10的国家 """

last = dataDic[15].copy(deep = True)
last['Test/Pop'] = last['totalTests'] / last['population']

# 将last中的数据按Test/Pop降序排序，空数据排在最后位
last.sort_values('Test/Pop', ascending = False, inplace = True, na_position = 'last')
print(last)

# 设置画图属性
fig = plt.figure(figsize = (10, 10))                                            # 窗口大小
fig.canvas.set_window_title('Test Proportion Top 10 Countries')                 # 窗口标题
ax = plt.axes()                                                                 # 创建坐标对象
plt.title('截止2020.12.15累计检测人数占国家总人口比例最高的10个国家')           # 柱状图标题
ax.set_xlabel('Ranking', family = 'Times New Roman')
ax.set_ylabel('Test / Pop', family = 'Times New Roman')


""" 绘制柱状图：展示比例 """

""" 设置x轴坐标标签 """
rank = [i for i in range(1, 11)]
proportion = last['Test/Pop'].tolist()[0 : 10]                  # 10个国家的比例
countries = last['country'].tolist()[0 : 10]                    # countries储存排名前10的国家名称
ax.set_xticks(rank)
ax.set_xticklabels(countries)
plt.xticks(rotation = 300)                                      # 使x坐标标签斜着，以防重叠
print(countries)
print(proportion)

""" 绘制10条不同颜色的柱状 """

drawBars(rank, proportion)
for a, b in zip(rank, proportion):
    """设置排名标签"""
    plt.text(a, b + 0.001, '{}%'.format(round(proportion[a - 1] * 100, 2)), family = 'Times New Roman',
             ha = 'center', va = 'bottom', fontsize = 10, rotation = -300)
plt.legend(countries, loc = 'upper right')


""" b、截止2020.12.15全球新冠疫情地图 """

last = dataDic[15].copy(deep = True)
countries = last['country'].tolist()                    # 国家名称
countries[countries.index('USA')] = 'United States'     # 地图中'USA'名为'United States'
cases = last['totalCases'].tolist()                     # 对应累计确诊人数
print(countries, cases)

map_visualmap(countries, cases).render("截止2020.12.15世界新冠疫情分布情况" + ".html")
print('###### Done! ######')




""" 9、针对全球累计确诊数，利用前 10 天采集到的数据做后 5 天的预测 """
print('\n\n--------------------------------------------9--------------------------------------------\n\n')



# 取前15天数据
y = worldSum[0 : 15]
print(y)

# 设置画图属性
fig = plt.figure(figsize = (10, 10))                                # 窗口大小
fig.canvas.set_window_title('Global Trends Predic')                 # 窗口标题
ax1 = plt.axes()

""" 设置x坐标标签 """

ax1.set_xticks([i for i in range(1, 16)])
xlabel = ['2020.12.{}'.format(i) for i in range(1, 16)]
ax1.set_xticklabels(xlabel, rotation = 300, fontsize = 8)
ax1.set_xlabel('日期')
ax1.set_ylabel('TotalCases')

# 绘制第1-15天的散点图
ax1.scatter([i for i in range(1, 16)], y, s = 15, color = 'r')
ax1.spines['right'].set_visible(False)                   # 隐藏右边框
ax1.spines['top'].set_visible(False)                     # 隐藏上边框
ax1.grid(True, linestyle = '--', alpha = 0.8)            # 设置网格线

""" 预测曲线 """


""" 线性回归 """

lm = linear_model.LinearRegression()
X = []
for i in range(1, 16):
    """ 获得训练数据X """
    x = []
    x.append(i)
    X.append(x)
model = lm.fit(X[0 : 10], y[0 : 10])
print(model.intercept_, model.coef_)                                # 拟合直线的斜率和y轴截距
score = model.score(X, y)
print(score)                                                        # 模型打分，高于95%则高度拟合
equation = 'y = {}x + {}'.format(round(model.coef_[0], 2), round(model.intercept_, 2))  # 拟合方程

# 绘制该直线
x = [i for i in range(1, 16)]
y2 = model.predict(X)
print(x, y2)
ax1.plot(x, y2, 'b--', alpha = 0.8)

model = lm.fit(X[10 : 16], y[10 : 16])
score_2 = model.score(X[10 : 16], y[10 : 16])
equation_2 = 'y = {}x + {}'.format(round(model.coef_[0], 2), round(model.intercept_, 2))  # 拟合方程

# 绘制
x = [i for i in range(11, 16)]
y2 = model.predict(X[10 : 16])
print(X[10 : 16], x, y2)
ax1.plot(x, y2, 'g--', alpha = 0.8)


""" 设置鼠标移动到点上时，显示点坐标（日期，PM） """
po_annotation = set_annotation()



ax1.set_title('2020年12.1-12.15世界新冠疫情变化情况预测分析')        # 图画标题
ax1.legend(['拟合曲线：{}，拟合度：{}'.format(equation, round(score, 2)),
            '拟合曲线：{}，拟合度：{}'.format(equation_2, round(score_2, 2))
            ], loc = 'upper left')    # 设置图例

on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)




""" 10、打分 """
print('\n\n--------------------------------------------10--------------------------------------------\n\n')



""" 
    打分标准：（截止12.15）   
                1、累计确诊病例占总人口比例
                2、累计检测数占总人口比例
                3、新增确诊数占总人口比例
                4、死亡人数占累计确诊数比例
"""

first = dataDic[1].copy(deep = True)      # 第1天数据
last = dataDic[15].copy(deep = True)      # 第15天数据

""" 将Taiwan和HongKong并入China再绘图 """
print(first.loc[first.country == 'China']['totalCases'])
first.loc[first.country == 'China']['totalCases'] += first.loc[first.country == 'Taiwan']['totalCases'] + first.loc[first.country == 'Hong Kong']['totalCases']

last.loc[last.country == 'China']['totalCases'] += last.loc[last.country == 'Taiwan']['totalCases'] + last.loc[last.country == 'Hong Kong']['totalCases']
last.loc[last.country == 'China']['totalTests'] += last.loc[last.country == 'Taiwan']['totalTests'] + last.loc[last.country == 'Hong Kong']['totalTests']
last.loc[last.country == 'China']['totalDeaths'] += last.loc[last.country == 'Taiwan']['totalDeaths'] + last.loc[last.country == 'Hong Kong']['totalDeaths']
last.loc[last.country == 'China']['population'] += last.loc[last.country == 'Taiwan']['population'] + last.loc[last.country == 'Hong Kong']['population']


""" 1、累计确诊病例占总人口比例排名 """
last['Case/Pop'] = last['totalCases'] / last['population']      # 新增列：为累计确诊占总人口比

# 将各国数据按确诊比例升序排序，且空数据国家排在最后
last.sort_values('Case/Pop', ascending = True, inplace = True, na_position = 'last')
countries = last['country'].tolist()
score = {}
for i in range(len(last)):
    """ 按排名给分，排名越靠后分值越高 """
    score[countries[i]] = i


""" 2、累计检测数占总人口比例 """
last['Test/Pop'] = last['totalTests'] / last['population']      # 新增列：为累计检测数占总人口比

# 将各国数据按检测比例升序排序，且空数据国家排在最后
last.sort_values('Test/Pop', ascending = True, inplace = True, na_position = 'last')
get_score(last)


""" 3、新增确诊比例排名 """
# 新增列"CumulDiag/Pop"：15日内新增确诊累计占总人口比
last['CumulDiag/Pop'] = (last['totalCases'] - first['totalCases']) / last['population']    

# 将各国数据按检测比例升序排序，且空数据国家排在最后
last.sort_values('CumulDiag/Pop', ascending = True, inplace = True, na_position = 'last')
get_score(last)


""" 4、死亡人数占累计确诊数比例 """
last['Death/Pop'] = last['totalDeaths'] / last['population']    # 新增列：为死亡人数占累计确诊人数比

# 将各国数据按检测比例升序排序，且空数据国家排在最后
last.sort_values('Death/Pop', ascending = True, inplace = True, na_position = 'last')
get_score(last)


""" 得到最终排名：得分越小说明应对越好 """
score = sorted(score.items(), key = lambda e : e[1], reverse = False)       # 以列表形式进行键值排序
score = dict(score)                                                         # 再将其转回字典类型
print(score)


""" 将Taiwan和HongKong并入China再绘图 """
score.pop('Taiwan')
score.pop('Hong Kong')
bestCountry = score.keys()
bestCountry = list(bestCountry)

""" 设置绘图属性 """

fig = plt.figure(figsize = (10, 10))                                        # 窗口大小
fig.canvas.set_window_title('Best 20 Countries In Defensing nCoV')          # 窗口标题
ax = plt.axes()                                                             # 创建坐标对象
ax.set_title('* 我认为应对新冠最好的20个国家 *')                            # 线图标题
ax.set_xlabel('Ranking', family = 'Times New Roman')
ax.set_ylabel('scores', family = 'Times New Roman')

""" 设置x轴坐标标签 """

rank = [i for i in range(1, 21)]        # 排名
top20 = list(score.values())[0 : 20]    # 排名前20的国家得分
ax.set_xticks(rank)
xlabel = bestCountry[0 : 20]            # 国家名称
ax.set_xticklabels(xlabel)
plt.xticks(rotation = 300)              # 使x坐标标签斜着，以防重叠
print(top20)
print(xlabel)

""" 绘制20条不同颜色的柱状 """

drawBars(rank, top20)
for a, b in zip(rank, top20):
    """设置排名标签"""
    plt.text(a, b + 2, top20[a - 1], family = 'Times New Roman',
             ha = 'center', va = 'bottom', fontsize = 10)
plt.legend(xlabel, loc = 'upper right', bbox_to_anchor = (1.133, 1), borderaxespad = 0.)

plt.show()

