# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

import csv

class NcovPipeline(object):
    def open_spider(self, spider):
        try:
            self.file = open('nCoV_All.csv', 'a', encoding = 'gbk', newline = '')
            #self.file = open('nCoV_15.csv', 'w', encoding = 'gbk', newline = '')
            self.csv = csv.writer(self.file)
            """设置表格抬头，方便数据处理"""
            #self.csv.writerow(['country', 'totalCases', 'newCases', 'totalDeaths', 'newDeaths', 'totalRecoverd', 'totalTests', 'population'])
        except Exception as e:
            print(e)

    def process_item(self, item, spider):
        self.csv.writerow(list(item.values()))
        return item

    def close_spider(self, spider):
        self.file.close()
