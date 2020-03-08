from scipy.optimize import curve_fit
import plotly.graph_objects as go
from uncertainties import ufloat
import plotly_express as px
import streamlit as st
import urllib.request
import pandas as pd
import numpy as np
import datetime
import re


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


def get_fit(func,slot,list_conf):
	best_fit_ab, covar = curve_fit(func, slot, list_conf)
	sigma_ab = np.sqrt(np.diagonal(covar))
	return best_fit_ab, sigma_ab


def plot_figure(func,list_conf,best_fit_ab,sigma_ab,slot,kind):
	a = ufloat(best_fit_ab[0], sigma_ab[0])
	b = ufloat(best_fit_ab[1], sigma_ab[1])

	bound_upper = func(slot, *(best_fit_ab + sigma_ab))
	bound_lower = func(slot, *(best_fit_ab - sigma_ab))

	text_res = "Best fit parameters on the " + str(max(slot)) + "th for " + kind + " :\na = {}\nb = {}".format(a, b)
	st.text(text_res)

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=slot, y=list_conf, mode='markers', name=kind,line_color='black'))
	fig.add_trace(go.Scatter(x=slot, y=bound_upper, fill='tonexty', mode='lines', line_color='grey',showlegend=False))
	fig.add_trace(go.Scatter(x=slot, y=bound_lower, fill='tonexty',mode='lines', name='Error',line_color='grey'))
	fig.add_trace(go.Scatter(x=slot, y=func(slot, *best_fit_ab), mode='lines', name='Fit Exp',line_color='red'))

	fig.update_layout(xaxis_title="day")

	predict = func(max(slot)+1, *best_fit_ab)
	st.error('Forecast for ' + str(max(slot)+1) + 'th day on ' + kind + ' = ' + str(int(round(predict))))

	return fig


def main():

	st.title("Statistic on COVID-19 in Italy")
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

	list_d = list_death[:lung_data]
	list_c = list_conf[:lung_data]
	list_r = list_rec[:lung_data]

	list_conf = [x for x in list_c if x != 0]
	#n_start = len(list_c)-len(list_conf)

	list_conf = list_c[30:]
	list_death = list_d[30:]
	list_rec = list_r[30:]
	list_dates = list_date[30:]

	dict_el = {'1-INFECTED' : list_conf,'2-RECOVERD' : list_rec,'3-DEATHS' : list_death}
	df = pd.DataFrame(dict_el, index = list_dates)
	st.text('')

	st.bar_chart(df)

	slot = np.arange(1, len(list_dates)+1)

	##fit Exponential
	st.subheader('Exponential fit : ')
	st.latex(r'''func = ae^{b(day)}''')

	func = lambda x, a, b: a * np.exp(b * x)

	##fit infection

	param_fit = get_fit(func,slot,list_conf)
	best_fit_ab = param_fit[0]
	sigma_ab = param_fit[1]

	fig = plot_figure(func,list_conf,best_fit_ab,sigma_ab,slot,'INFECTION')
	st.plotly_chart(fig)

	##fit death

	param_fit = get_fit(func,slot,list_death)
	best_fit_ab = param_fit[0]
	sigma_ab = param_fit[1]

	fig = plot_figure(func,list_death,best_fit_ab,sigma_ab,slot,'DEATH')
	st.plotly_chart(fig)

	##fit recovered

	param_fit = get_fit(func,slot,list_rec)
	best_fit_ab = param_fit[0]
	sigma_ab = param_fit[1]

	fig = plot_figure(func,list_rec,best_fit_ab,sigma_ab,slot,'RECOVERED')
	st.plotly_chart(fig)



if __name__ == '__main__':
	main()