import streamlit as st
import urllib.request
import pandas as pd
import datetime
import re

def get_death(file):
	url_death = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/' \
				+ file
	with urllib.request.urlopen(url_death) as testfile, open('dataset.csv', 'w') as f:
		f.write(testfile.read().decode())

	f = open("dataset.csv", "r")
	xml=f.read()
	date = re.findall(r'.*<th>([\s+<th>\d+\/\d+\/\d+<\/th>]+)\s+<\/tr>', xml)
	num_d = re.findall(r'.*<td>Italy<\/td>([\s+<td>\d+<\/td>]+)\s+<\/tr>',xml)
	list_date = re.findall(r'(\d+\/\d+\/\d+)',date[0])
	list_death = re.findall(r'\d+',num_d[0])[2:]

def get_date_num(file):
	url_death = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/' \
				+ file
	with urllib.request.urlopen(url_death) as testfile, open('dataset.csv', 'w') as f:
		f.write(testfile.read().decode())

	f = open("dataset.csv", "r")
	xml=f.read()
	date = re.findall(r'.*<th>([\s+<th>\d+\/\d+\/\d+<\/th>]+)\s+<\/tr>', xml)
	num = re.findall(r'.*<td>Italy<\/td>([\s+<td>\d+<\/td>]+)\s+<\/tr>',xml)
	list_dat = re.findall(r'(\d+\/\d+\/\d+)',date[0])
	list_n = re.findall(r'\d+',num[0])[2:]
	list_date = [datetime.datetime.strptime(x, '%m/%d/%y').strftime('%m/%d/%y') for x in list_dat]
	list_num = [int(x) for x in list_n]
	return list_date, list_num


def main():

	st.title("Statistic on COVID-19")
	st.subheader("data from : https://github.com/CSSEGISandData/COVID-19")
	list_date1 = get_date_num('time_series_19-covid-Deaths.csv')[0]
	list_death = get_date_num('time_series_19-covid-Deaths.csv')[1]
	list_date2 = get_date_num('time_series_19-covid-Confirmed.csv')[0]
	list_conf = get_date_num('time_series_19-covid-Confirmed.csv')[1]
	list_date3 = get_date_num('time_series_19-covid-Recovered.csv')[0]
	list_rec = get_date_num('time_series_19-covid-Recovered.csv')[1]

	list_len_date=[len(list_date1),len(list_date2),len(list_date3)]
	list_li_date=[list_date1,list_date2,list_date3]
	el_min = min([len(list_date1),len(list_date2),len(list_date3)])
	pos = list_len_date.index(el_min)
	list_date = list_li_date[pos]
	lung_data = len(list_date)

	list_death = list_death[:lung_data]
	list_conf = list_conf[:lung_data]
	list_rec = list_rec[:lung_data]

	dict_el = {'INFECTED':list_conf,'RECOVERD':list_rec,'DEATHS':list_death}
	df = pd.DataFrame(dict_el, index = list_date)

	st.bar_chart(df)










if __name__ == '__main__':
	main()