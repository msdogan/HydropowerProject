# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:39:23 2016

@author: msdogan
"""
# web scraping module - Mustafa Dogan
# note: you cannot request hourly data from OASIS more than 31 days
import requests, zipfile, StringIO

example_url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20160919T00:00-0000&enddatetime=20160920T00:00-0000&version=1&market_run_id=DAM&grp_type=ALL_APNODES&resultformat=6'
example_url2 = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20130919T07:00-0000&enddatetime=20130920T07:00-0000&version=1&market_run_id=DAM&node=LAPLMG1_7_B2&resultformat=6'

# building a url - components
queryname = 'PRC_LMP' # download locational marginal price
market_run_id = 'DAM' # day ahead market
grp_type = 'ALL_APNODES' # you can also put node names - Note: file size gets too big if you download all for a month so api does not allow that
resultformat = '6' # this downloads as csv
api_name = 'http://oasis.caiso.com/oasisapi/SingleZip?'
node = 'LAPLMG1_7_B2' # node location

n_days = {'Jan':31, 'Feb':28, 'Mar':31, 'Apr':30, 'May':31,  'Jun':30, 'Jul':31, 'Aug':31, 'Sep':30, 'Oct':31, 'Nov':30, 'Dec':31}
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']

# enter starting and ending year and month that you want to download
# the code below downloads all hourly price data for months you specified
s_year = 2015 # starting year
e_year = 2016 # ending year
s_month = 9 # starting month
e_month = 9 # ending month

# let the code do its magic
t = 0
y = s_year
m = s_month
i = s_month
while t <= ((e_year-s_year)*12 + (e_month-s_month)):
    if i < 10:
        cm = '0' + str(i)
    else:
        cm = i
    startdatetime = str(str(y) + str(cm) + '01' + 'T00:00-0000') # start time
    enddatetime = str(str(y) + str(cm) + str(n_days[months[i-1]]) + 'T00:00-0000') # end time
#    url = api_name + 'queryname=' + queryname + '&' + 'startdatetime=' + startdatetime + '&' + 'enddatetime=' + enddatetime + '&' + 'version=1' + '&' +  'market_run_id=' + market_run_id + '&' + 'grp_type=' + grp_type + '&' + 'resultformat=' + resultformat
    url = api_name + 'queryname=' + queryname + '&' + 'startdatetime=' + startdatetime + '&' + 'enddatetime=' + enddatetime + '&' + 'version=1' + '&' +  'market_run_id=' + market_run_id + '&' + 'node=' + node + '&' + 'resultformat=' + resultformat
    print('now downloading: ' + startdatetime, enddatetime)    
    r = requests.get(url, stream=True) # request price data
    z = zipfile.ZipFile(StringIO.StringIO(r.content)) # url request returns a zip file
    z.extractall('Z:\HydropowerProject\Pumped_Storage\Price_Data') # unzip files, you can also specify a directory  
    if (m % 12) == 0: # happy new year!
        y += 1 # next year
        i = 0 # reset months
    m += 1
    t += 1
    i += 1