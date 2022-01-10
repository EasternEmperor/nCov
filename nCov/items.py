# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NcovItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    date = scrapy.Field()           # 时间
    country = scrapy.Field()        # 国家名字
    totalCases = scrapy.Field()     # 累计确诊
    newCases = scrapy.Field()       # 新增确诊
    totalDeaths = scrapy.Field()    # 累计死亡
    newDeaths = scrapy.Field()      # 新增死亡
    totalRecoverd = scrapy.Field()  # 累计痊愈
    totalTests = scrapy.Field()     # 累计检测人数
    population = scrapy.Field()     # 国家总人口