# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:39:23 2016

@author: msdogan
"""
# web scraping module
import requests, zipfile, StringIO

example_url = 'http://oasis.caiso.com/oasisapi/SingleZip?queryname=PRC_LMP&startdatetime=20160919T07:00-0000&enddatetime=20160920T07:00-0000&version=1&market_run_id=DAM&grp_type=ALL_APNODES&resultformat=6'

# building a url
queryname = 'PRC_LMP'
startdatetime = '20160919T07:00-0000' # GMT format: Year Month Day T Hour
enddatetime = '20160920T07:00-0000'
market_run_id = 'DAM'
grp_type = 'ALL_APNODES' # you can also put node names
resultformat = 6 # this downloads as csv

# download hourly price data and save it to specified directory
r = requests.get(example_url, stream=True)
z = zipfile.ZipFile(StringIO.StringIO(r.content))
z.extractall() # you can also speciyf a directory
