# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:39:23 2016

@author: msdogan
"""
# web scraping module - Mustafa Dogan
# note: you cannot request hourly data from OASIS more than 31 days
import requests, zipfile, StringIO

example_url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_HASP_LMP&startdatetime=20130919T07:00-0000&enddatetime=20130919T08:00-0000&version=2&market_run_id=HASP&node=LAPLMG1_7_B2&resultformat=6'
example_url2 = 'http://oasis.caiso.com/oasisapi/GroupZip?groupid=HASP_LMP_GRP&startdatetime=20130901T07:00-0000&enddatetime=20130919T07:00-0000&version=1&resultformat=6' # this returns only 1 hour data starting start data. End date does not matter

# building a url - components
queryname = 'PRC_HASP_LMP' # download marginal price
market_run_id = 'HASP' 
group_id = 'HASP_LMP_GRP'
grp_type = 'ALL_APNODES' # you can also put node names - Note: file size gets too big if you download all for a month so api does not allow that
resultformat = '6' # this downloads as csv
single_api_name = 'http://oasis.caiso.com/oasisapi/SingleZip?'
group_api_name = 'http://oasis.caiso.com/oasisapi/GroupZip?'
node = ['LAPLMG1_7_B2'] # enter node names you want to download
# look at node names from http://www.caiso.com/pages/pricemaps.aspx

# group url building. returns only 1 hour of data
#url_group = group_api_name + 'groupid=' + group_id + '&' + 'startdatetime=' + startdatetime + '&' + 'enddatetime=' + enddatetime + '&' + 'version=1' + '&' + 'grp_type=' + grp_type + '&' + 'resultformat=' + resultformat

n_days = {'Jan':31, 'Feb':28, 'Mar':31, 'Apr':30, 'May':31,  'Jun':30, 'Jul':31, 'Aug':31, 'Sep':30, 'Oct':31, 'Nov':30, 'Dec':31}
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']

# enter starting and ending year and month that you want to download
# the code below downloads all hourly price data for months you specified
s_year = 2013 # starting year
e_year = 2013 # ending year
s_month = 9 # starting month
e_month = 9 # ending month

# let the code do its magic
t = 0
y = s_year # beginning year
m = s_month # beginning month
i = s_month
ds = ['01', '11', '21'] # divide a month into 10 days - starting day
de = ['11', '21', '01'] # divide a month into 10 days - ending day
for n in range(len(node)):
    while t <= ((e_year-s_year)*12+(e_month-s_month)): # retrieve 10 day period
        if i < 10:
            sm = '0' + str(i) # starting month <10
            em = '0' + str(i+1) # ending month <10
            if i == 9:
                 em = i + 1
        else:
            sm = i # starting month >10
            em = i + 1 # ending month >10
        for x,day in enumerate(ds):
            if x <= 1:
                emm = sm
            else:
                emm = em
            startdatetime = str(str(y) + str(sm) + str(ds[x]) + 'T07:00-0000') # start time 
            enddatetime = str(str(y) + str(emm) + str(de[x]) + 'T07:00-0000') # end time 
            url_single = single_api_name+'queryname='+queryname +'&'+'startdatetime='+startdatetime+'&'+'enddatetime='+enddatetime +'&'+'version=1'+'&'+'market_run_id='+market_run_id +'&'+'node='+node[n]+'&'+'resultformat='+resultformat
            status = 0 
            print('now downloading: ' + startdatetime, enddatetime)
            r = requests.get(url_single, stream=True, verify=False, timeout=500) # request price data, single or all
            z = zipfile.ZipFile(StringIO.StringIO(r.content)) # url request returns a zip file
            z.extractall('Z:/Price_Data') # unzip files, you can also specify a directory 
        if (m % 12) == 0: # happy new year!
            y += 1 # next year
            i = 0 # reset months
        m += 1
        t += 1
        i += 1