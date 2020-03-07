import streamlit as st
import pandas as pd

def main():

	st.title("Statistic on COVID-19")
	st.subheader("data from : https://github.com/CSSEGISandData/COVID-19")

	url = 'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv'
	df = pd.read_csv(url, error_bad_lines=False)
	df




if __name__ == '__main__':
	main()