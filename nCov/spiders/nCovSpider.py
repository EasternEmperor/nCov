import scrapy
from nCov.items import NcovItem
import csv
import re

class MySpider(scrapy.Spider):
    name = 'nCov'
    allowed_domains = ['worldometers.info']
    start_urls = ['https://www.worldometers.info/coronavirus/#countries']

    def parse(self, response):
        item = NcovItem()
        for each in response.xpath("//*[@id='main_table_countries_yesterday']/tbody[1]/*"):
            """爬取该路径下的所有国家疫情信息"""
            """日期"""
            item['date'] = response.xpath("//*[@id='news_block']/div[1]/h4/text()").extract()[0].split()[1]

            """国家名字"""
            coun = each.xpath("./td[2]/a/text()").extract()
            if len(coun) != 0:
                item['country'] = coun[0]
            else:
                item['country'] = ''

            """累计确诊"""
            ttcases = each.xpath("./td[3]/text()").extract()
            if len(ttcases) != 0:
                item['totalCases'] = ttcases[0].replace(',', '')     # 去掉数字间的逗号
            else:
                item['totalCases'] = ''

            """新增确诊"""
            ncases = each.xpath("./td[4]/text()").extract()
            if len(ncases) != 0:
                item['newCases'] = ncases[0].split('+')[1].replace(',', '')     # 去掉数字间的逗号和数字前的'+'号
            else:
                item['newCases'] = ''

            """累计死亡"""
            ttdeaths = each.xpath("./td[5]/text()").extract()
            if len(ttdeaths) != 0:
                item['totalDeaths'] = ttdeaths[0].replace(',', '').strip()      # 去掉数字间的逗号
            else:
                item['totalDeaths'] = ''

            """新增死亡"""
            ndeaths = each.xpath("./td[6]/text()").extract()
            if len(ndeaths) != 0:
                item['newDeaths'] = ndeaths[0].split('+')[1].replace(',', '')   # 去掉数字间的逗号
            else:
                item['newDeaths'] = ''

            """累计痊愈"""
            ttrecovered = each.xpath("./td[7]/text()").extract()
            if len(ttrecovered) != 0:
                item['totalRecoverd'] = ttrecovered[0].replace(',', '')     # 去掉数字间的逗号
            else:
                item['totalRecoverd'] = ''

            """检测总人数"""
            tttests = each.xpath("./td[13]/text()").extract()
            if len(tttests) != 0:
                item['totalTests'] = tttests[0].replace(',', '')     # 去掉数字间的逗号
            else:
                item['totalTests'] = ''

            """国家总人口"""
            pop = each.xpath("./td[15]/a/text()").extract()
            if len(pop) != 0:
                item['population'] = pop[0].replace(',', '')     # 去掉数字间的逗号
            else:
                item['population'] = ''

            if (item['country']):
                """去除国家名为空的数据"""
                yield(item)         # 将item返回到pipeline模块
            else:
                print('-----------USELESS DATA-------------', item['country'])
