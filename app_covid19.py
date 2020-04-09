from scipy.optimize import curve_fit
import plotly.graph_objects as go
from uncertainties import ufloat
import streamlit as st
import urllib.request
import pandas as pd
import numpy as np
import datetime
import re


def get_date_num(feature):
    df = pd.read_csv(
        'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/'
        'dpc-covid19-ita-andamento-nazionale.csv',sep=',')

    list_dat = []
    for el_data in df['data'].values:
        el_date = re.sub(r'(\d{4}-\d{2}-\d{2}).*', r'\1', el_data)
        list_dat.append(el_date)

    list_dat.insert(0, '2020-02-23')
    list_dat.insert(0, '2020-02-22')
    list_date = [datetime.datetime.strptime(x, '%Y-%m-%d').strftime('%Y-%m-%d') for x in list_dat]

    list_num = list(df[feature].values)

    #with urllib.request.urlopen(url_death) as testfile, open('dataset.csv', 'w') as f:
    #    f.write(testfile.read().decode())

    #f = open("dataset.csv", "r")
    #xml = f.read()
   # date = re.findall(r'.*<th>([\s+<th>\d+\/\d+\/\d+<\/th>]+)\s+<\/tr>', xml)
    #num = re.findall(r'.*<td>Italy<\/td>([\s+<td>\d+<\/td>]+)\s+<\/tr>', xml)
    #list_dat = re.findall(r'(\d+\/\d+\/\d+)', date[0])
    #list_n = re.findall(r'\d+', num[0])[2:]
    #
    #list_num = [int(x) for x in list_n]
    return list_date, list_num


def get_fit(func, slot, list_conf):
    best_fit_ab, covar = curve_fit(func, slot, list_conf)
    sigma_ab = np.sqrt(np.diagonal(covar))
    return best_fit_ab, sigma_ab


def plot_figure(func, list_conf, best_fit_a_b, sigma_a_b, slot, kind, option):
    a = ufloat(best_fit_a_b[0], sigma_a_b[0])
    b = ufloat(best_fit_a_b[1], sigma_a_b[1])

    bound_upper = func(slot, *(best_fit_a_b + sigma_a_b))
    bound_lower = func(slot, *(best_fit_a_b - sigma_a_b))

    text_res = "Best fit parameters on the " + str(max(slot)) + "th for " + kind + " :\na = {}\nb = {}".format(a, b)
    st.text(text_res)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=slot, y=list_conf, mode='markers', name=kind, line_color='black'))
    #fig.add_trace(go.Scatter(x=slot, y=bound_upper, mode='lines', line_color='grey', showlegend=False))
    #fig.add_trace(go.Scatter(x=slot, y=bound_lower, fill='tonexty', mode='lines', name='Error', line_color='grey'))
    fig.add_trace(go.Scatter(x=slot, y=func(slot, *best_fit_a_b), mode='lines', name='Fit ' + option, line_color='red'))

    fig.update_layout( xaxis_title="days", xaxis = dict(
    tickmode = 'array',
    tickvals = slot,
    ticktext = [str(x)+'.02' for x in range(22,30)]+[str(x)+'.03' for x in range(1,32)]+[str(x)+'.04' for x in range(1,31)]))

    predict = func(max(slot) + 1, *best_fit_a_b)
    st.error('Forecast for ' + str(max(slot) + 1) + 'th day on ' + kind + ' = ' + str(int(round(predict))))

    return fig


def main():
    st.title("Statistic on COVID-19 in Italy")
    st.subheader("data from : https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/"
                 "dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv")
    ##list of three features
    list_dates = get_date_num('data')[0]
    list_death = get_date_num('deceduti')[1]
    list_death.insert(0, 3)
    list_death.insert(0, 2)
    list_conf2 = get_date_num('totale_casi')[1]
    list_conf2.insert(0, 155)
    list_conf2.insert(0, 62)
    list_rec = get_date_num('dimessi_guariti')[1]
    list_rec.insert(0, 2)
    list_rec.insert(0, 1)



    #list_len_date = [len(list_date1), len(list_date2), len(list_date3)]
    #list_li_date = [list_date1, list_date2, list_date3]
    #el_min = min([len(list_date1), len(list_date2), len(list_date3)])
    #pos = list_len_date.index(el_min)
    #list_date = list_li_date[pos]
    #lung_data = len(list_date)

    #list_d = list_death[:lung_data]
    #list_c = list_conf[:lung_data]
    #list_r = list_rec[:lung_data]

    #list_conf = [x for x in list_c if x != 0]
    # n_start = len(list_c)-len(list_conf)

    #list_conf2 = list_c[31:]
    #list_death = list_d[31:]
    #list_rec = list_r[31:]
    #list_dates = list_date[31:]

    list_conf1 = [x1 - x2 for (x1, x2) in zip(list_conf2, list_rec)]
    list_conf = [x1 - x2 for (x1, x2) in zip(list_conf1, list_death)]

    dict_el = {'1-INFECTED': list_conf, '2-RECOVERD': list_rec, '3-DEATHS': list_death}
    df = pd.DataFrame(dict_el, index=list_dates)
    st.text('')

    st.bar_chart(df)

    delta_conf = [x2-x1 for x1,x2 in zip(list_conf[:len(list_conf)-1],list_conf[1:])]

    slot = np.arange(1, len(list_dates)+1)


    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=slot, y=delta_conf, mode='markers', showlegend=False,line_color='red'))
    #fig.add_trace(go.Scatter(x=slot, y=delta_conf, mode='lines',showlegend=False, line_color='red'))
    #fig.update_layout( xaxis_title="days", xaxis = dict(
    #tickmode = 'array',
    #tickvals = slot,
    #ticktext = [str(x)+'.02' for x in range(23,30)]+[str(x)+'.03' for x in range(1,32)]+[str(x)+'.04' for x in range(1,31)]), yaxis_title="Delta Positivi")

    #st.plotly_chart(fig)
    
    
    delta_d=[list_death[x+1]-list_death[x] for x in range(len(list_death)-1)]

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x=slot, y=delta_d, mode='markers', showlegend=False,line_color='black'))
    #fig.add_trace(go.Scatter(x=slot, y=delta_d, mode='lines',showlegend=False, line_color='black'))
    #fig.update_layout( xaxis_title="days", xaxis = dict(
    #tickmode = 'array',
    #tickvals = slot,
    #ticktext = [str(x)+'.02' for x in range(23,30)]+[str(x)+'.03' for x in range(1,32)]+[str(x)+'.04' for x in range(1,31)]), yaxis_title="Delta Dead")
    #st.plotly_chart(fig)

    
    fig = go.Figure(data=[
        go.Bar(name='Delta_Positives', x=list_dates[1:], y=delta_conf),
        go.Bar(name='Delta_Deaths', x=list_dates[1:], y=delta_d)
    ])
    fig.update_layout(barmode='group')
    st.plotly_chart(fig)
    
    
    slot1=np.array([x+2 for x in slot])

    ####################################################################################
    #func = lambda x, a, b: a * np.exp(b * (x - 1))
    list_tamponi = get_date_num('tamponi')[1]
    #param_fit_tamponi = get_fit(func, slot, list_tamponi)
    #best_fit_ab_tamponi = param_fit_tamponi[0]
    #sigma_ab_tamponi = param_fit_tamponi[1]

    #####
    func = lambda x, a, b: a*x+b
    param_fit_tam = get_fit(func, slot1[15:-2], list_tamponi[15:])
    best_fit_ab_tam = param_fit_tam[0]
    #####

#    best_fit_ab_tam
#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=slot1, y=list_tamponi, mode='markers',name='Tamponi' ,line_color='orange'))
#    fig.add_trace(go.Scatter(x=slot, y=list_conf, mode='markers', name='Positivi',line_color='black'))
#    fig.add_trace(go.Scatter(x=slot1[15:-2], y=func(slot1[15:-2], *best_fit_ab_tam), name='Linear Fit',
#                             line=dict(color='blue', width=1, dash='dash'), line_color='blue'))
#    fig.update_layout( xaxis_title="days", xaxis = dict(
#    tickmode = 'array',
#    tickvals = slot1,
#    ticktext = [str(x)+'.02' for x in range(24,30)]+[str(x)+'.03' for x in range(1,32)]+[str(x)+'.04' for x in range(1,31)]))
#    st.plotly_chart(fig)

    ##Fit Ratio
    #func = lambda x, a, b: a*(x**b)
    #func = lambda x, a, b: c * np.exp(b * (x)) / (1 - (c / a) * (1 - np.exp(b * (x))))
    func = lambda x, a, b: a*x+b
    ratio = [1.0 * x1 / x2 for x1, x2 in zip(list_conf[2:], list_tamponi)]
    param_fit_ratio = get_fit(func, slot[3:], ratio[1:])
    best_fit_ab_ratio = param_fit_ratio[0]
    sigma_ab_ratio = param_fit_ratio[1]


 #   bound_upper = func(slot[3:], *(best_fit_ab_ratio + sigma_ab_ratio))
 #   bound_lower = func(slot[3:], *(best_fit_ab_ratio - sigma_ab_ratio))
 #   best_fit_ab_ratio
 #   fig = go.Figure()
#    fig.add_trace(go.Scatter(x=slot[3:], y=ratio[1:], mode='markers', name='Positivi/Tamponi',line_color='red'))
#    fig.add_trace(go.Scatter(x=slot[3:], y=bound_upper, mode='lines', line_color='grey', showlegend=False))
#    fig.add_trace(go.Scatter(x=slot[3:], y=bound_lower, fill='tonexty', mode='lines', name='Error', line_color='grey'))
#    fig.add_trace(go.Scatter(x=slot[3:], y=func(slot[3:], *best_fit_ab_ratio), mode='lines',
#                             name=str(round(best_fit_ab_ratio[0],3))+'*x + '+str(round(best_fit_ab_ratio[1],3)),line_color='red'))
#    fig.update_layout( xaxis_title="day from 22.02" , yaxis_title='Ratio Infected/Swabs')
#    st.plotly_chart(fig)


    ##Tamponi
#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=slot1[15:-2], y=list_tamponi[15:], mode='markers',marker=dict(size=[12]*len(slot1[15:-2])),
#                             name='Tamponi' ,line_color='orange'))
#    fig.add_trace(go.Scatter(x=slot1[15:-2], y=func(slot1[15:-2], 12795, -170197),
#                             line=dict(color='blue', width=2, dash='dash'), name='Linear Fit'))
#    fig.update_layout(xaxis_title="days form 10.03" , xaxis = dict(
#        tickmode = 'array',
#        tickvals = slot1[15:-2],
#        ticktext = [str(x)+'.03' for x in range(10,32)]+[str(x)+'.04' for x in range(1,31)]) ,
#                      yaxis_title='Swabs')
#    st.plotly_chart(fig)

    ##RATIO

    func = lambda x, a, b: a*x+b
    param_fit_ratio1 = get_fit(func, slot[17:21], ratio[15:19])
    best_fit_ab_ratio1 = param_fit_ratio1[0]

    param_fit_ratio2 = get_fit(func, slot[21:-1], ratio[19:-1])
    best_fit_ab_ratio2 = param_fit_ratio2[0]

#    st.subheader('I believe that the true analysis cannot be made on the absolute Infected number. '
#            'The analysis must be done in terms of the ratio of Positives to Swab, therefore as a relative number. '
#            'As can be seen, the trends of this ratio grow linearly according to the days domain, '
#            'changing slope. Declining slopes are welcome. The number of tampons increases linearly in these days,'
#                 ' as can be verified,'
#            'therefore the fact that the slope of the ratio decreases is a positive thing. '
#            'The problem is the jumps! It is evident in the graphs how frequent they are. '
#            'This in my opinion is symptomatic of various factors. Delay of swab results, '
#            'distorted results of previous swabs and/or sudden increase in epidemic, '
#                 'due to the incubation period of the disease.')

#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=slot[17:], y=ratio[15:], mode='markers',marker=dict(size=[14]*len(slot[17:])) ,
#                             name='Positivi/Tamponi',line_color='red'))
#    fig.add_trace(go.Scatter(x=slot[17:], y=func(slot[17:], *best_fit_ab_ratio1),
#                             line=dict(color='black', width=2.5, dash='dash'), name=str(round(best_fit_ab_ratio1[0], 3))
#                                            + '*x + ' + str(round(best_fit_ab_ratio1[1], 3))))
#    fig.add_trace(go.Scatter(x=slot[21:], y=func(slot[21:], *best_fit_ab_ratio2),
#                             line=dict(color='blue', width=2.5, dash='dash'), name=str(round(best_fit_ab_ratio2[0], 3))
#                                            + '*x + ' + str(round(best_fit_ab_ratio2[1], 3))))
#
#    fig.update_layout( xaxis_title="days from 10.03", xaxis = dict(
#        tickmode = 'array',
#        tickvals = slot[17:],
#        ticktext = [str(x)+'.03' for x in range(10,32)]+[str(x)+'.04' for x in range(1,31)]) ,
#                       yaxis_title='Ratio Infected/Swabs')
#    st.plotly_chart(fig)

    fig = go.Figure([go.Bar(name='Swabs', marker=dict(color='orange'),x=list_dates[17:], y=list_tamponi[15:])])
    fig.update_layout(yaxis_title='Swabs')
    st.plotly_chart(fig)


    fig = go.Figure([go.Bar( marker=dict(color='red'),x=list_dates[8:], y=ratio[6:])])
    fig.update_layout(yaxis_title='Ratio: Positivies/Swabs')
    st.plotly_chart(fig)

    ##SQUARE
#    func = lambda x, a, b, c : a*x**2 + b*x + c
#    param_fit_ratio1 = get_fit(func, slot[12:21], list_conf[12:21])
#    best_fit_ab_ratio1 = param_fit_ratio1[0]

#    param_fit_ratio2 = get_fit(func, slot[21:-1], list_conf[21:-1])
#    best_fit_ab_ratio2 = param_fit_ratio2[0]

#    slot11 = np.append(slot[21:],np.array([26,27,28,29,30]))

#    st.text('fit dashed black')
#    best_fit_ab_ratio1
#    st.text('fit dashed blue')
#    best_fit_ab_ratio2

#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=slot[10:], y=list_conf[10:], mode='markers',marker=dict(size=[14]*len(slot[10:])) ,
#                             name='Positivi',line_color='red'))
#    fig.add_trace(go.Scatter(x=slot[10:], y=func(slot[10:], *best_fit_ab_ratio1),
#                             line=dict(color='black', width=2.5, dash='dash'), name='Square Fit'))
#    fig.add_trace(go.Scatter(x=slot11, y=func(slot11, *best_fit_ab_ratio2),
#                            mode='lines',line_color='blue', name='Square Fit'))

#    fig.update_layout( xaxis_title="day", xaxis = dict(
#        tickmode = 'array',
#        tickvals = slot[10:],
#        ticktext = ['03.03','04.03','05.03','06.03','07.03','08.03','09.03', '10.03', '11.03', '12.03', '13.03',
#                    '14.03','15.03','16.03','17.03','18.03','19.03','20.03']) ,
#                       yaxis_title='Infected')
#    st.plotly_chart(fig)


    ###############################################################################



    func = lambda x, a, b: a * np.exp(b * (x - 1))
    list_terapia = get_date_num('terapia_intensiva')[1]
    list_ricoverati = get_date_num('ricoverati_con_sintomi')[1]
    param_fit_ricoveri = get_fit(func, slot[2:], list_ricoverati)
    best_fit_ab_ricoveri = param_fit_ricoveri[0]
    sigma_ab_ricoveri = param_fit_ricoveri[1]
    ratio = [1.0*x1/x2 for x1, x2 in zip(list_death, list_terapia)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=slot, y=list_ricoverati, mode='markers', name='Ricoverati',line_color='black'))
#    fig.add_trace(go.Scatter(x=slot, y=func(slot[2:], *best_fit_ab_ricoveri), mode='lines', name='Exp',
#                             line_color='black'))
    fig.add_trace(go.Scatter(x=slot, y=list_terapia, mode='markers',name='Terapia Intensiva' ,line_color='red'))
    fig.add_trace(go.Scatter(x=slot, y=list_death[2:], mode='markers', name='Deceduti', line_color='grey'))
    #fig.add_trace(go.Scatter(x=slot, y=ratio, mode='lines', name='Ratio Death/Terapia_Intensiva', line_color='blue'))    
    fig.update_layout( xaxis_title="days", xaxis = dict(
    tickmode = 'array',
    tickvals = slot,
    ticktext = [str(x)+'.02' for x in range(24,30)]+[str(x)+'.03' for x in range(1,32)]+[str(x)+'.04' for x in range(1,31)]))
    
    st.plotly_chart(fig)





    slot = np.arange(1, len(list_dates) + 1)

    option = st.selectbox(
        'Choose the Epidemiologic Law :',
        ['Exponential Law', 'Gompertz Law', 'Logistic Law'])

    if option == 'Exponential Law':
        st.subheader('Exponential fit : ')
        st.latex(r'''func = ae^{b(day)}''')

        func = lambda x, a, b: a * np.exp(b * (x - 1))
    elif option == 'Gompertz Law':
        st.subheader('Gompertz fit : ')
        st.latex(r'''func = a*exp\left\{ ln\left(\frac{N(day=1)}{a} \right) e^{-b(day)} \right\}''')
        c = 1
        func = lambda x, a, b: a * np.exp(np.log(c / a) * np.exp(-b * (x)))
    else:
        st.subheader('Logistic fit : ')
        st.latex(r'''func = N(day = 1) \frac{e^{b(day)}}{1-\frac{N(day=1)}{a}\left[ 1-e^{b(day)}\right]}''')
        c = 1
        func = lambda x, a, b: c * np.exp(b * (x)) / (1 - (c / a) * (1 - np.exp(b * (x))))


    st.text('The days refer from 22/02/2020, which is day zero')

    ##fit infection
    c = list_conf[0]
    param_fit_inf = get_fit(func, slot, list_conf)
    best_fit_ab_inf = param_fit_inf[0]
    sigma_ab_inf = param_fit_inf[1]

        ##add compare 13.03
    import datetime
    x = datetime.datetime.now()
    d = int(x.strftime("%d"))
    param_fit_inf0 = get_fit(func, slot[:len(slot) - (d-3)], list_conf[:len(list_conf) - (d-3)])
    best_fit_ab_inf0 = param_fit_inf0[0]



    if sigma_ab_inf[0]/best_fit_ab_inf[0] < 0.3 or sigma_ab_inf[1]/best_fit_ab_inf[1] < 0.3:
        fig = plot_figure(func, list_conf, best_fit_ab_inf, sigma_ab_inf, slot, 'INFECTION', option)
        fig.add_trace(
            go.Scatter(x=np.array(slot[:len(slot)]), y=func(np.array(slot[:len(slot)]), *best_fit_ab_inf0),
                       line=dict(color='black', width=2, dash='dash'), name='fit '+ str(d-3)+' days before',
                       line_color='blue'))

        st.plotly_chart(fig)

    ##fit death
    c = list_death[2]
    param_fit_d = get_fit(func, slot, list_death)
    best_fit_ab_d = param_fit_d[0]
    sigma_ab_d = param_fit_d[1]

        ##compare 13.03
    param_fit_d0 = get_fit(func, slot[:len(slot) - (d-3)], list_death[:len(list_death) - (d-3)])
    best_fit_ab_d0 = param_fit_d0[0]

    if sigma_ab_d[0]/best_fit_ab_d[0] < 0.3 and sigma_ab_d[1]/best_fit_ab_d[1] < 0.3:
        fig = plot_figure(func, list_death, best_fit_ab_d, sigma_ab_d, slot, 'DEATH', option)
        fig.add_trace(
            go.Scatter(x=np.array(slot[:len(slot)]), y=func(np.array(slot[:len(slot)]), *best_fit_ab_d0),
                       line=dict(color='black', width=2, dash='dash'), name='fit '+ str(d-3)+' days before',
                       line_color='blue'))
        st.plotly_chart(fig)

    ##fit recovered
    c = list_rec[0]
    param_fit = get_fit(func, slot, list_rec)
    best_fit_ab = param_fit[0]
    sigma_ab = param_fit[1]
    if sigma_ab[0]/best_fit_ab[0] < 0.3 or sigma_ab[1]/best_fit_ab[1] < 0.3:
        fig = plot_figure(func, list_rec, best_fit_ab, sigma_ab, slot, 'RECOVERED', option)
        st.plotly_chart(fig)

    st.subheader('Peak Prediction and Arrest for the Infection ' + option)

    aa = 10
    bb = 100
    c = list_conf[0]
    slot_ricors = slot
    if option == 'Gompertz Law' or option == 'Logistic Law':
        while (bb - aa > 20):
            slot_aa = slot_ricors
            slot_bb = np.append(slot_ricors, max(slot_ricors) + 1)
            aa = func(slot_aa, *best_fit_ab_inf)[-1]
            bb = func(slot_bb, *best_fit_ab_inf)[-1]
            slot_ricors = np.append(slot_ricors, max(slot_ricors) + 1)
        list_y = func(slot_ricors, *best_fit_ab_inf)
        bound_upper = func(slot_ricors, *(best_fit_ab_inf + sigma_ab_inf))
        bound_lower = func(slot_ricors, *(best_fit_ab_inf - sigma_ab_inf))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=slot, y=list_conf, mode='markers', name='INFECTION', line_color='black'))
        fig.add_trace(go.Scatter(x=slot_ricors, y=list_y,
                                 mode='lines', name='Fit ' + option, line_color='red'))
        fig.add_trace(go.Scatter(x=slot_ricors, y=func(slot_ricors, *best_fit_ab_inf0),
                                 line=dict(color='black', width=2, dash='dash'), name='Fit '+str(d-3)+' days before',
                                 line_color='blue'))
        fig.add_trace(go.Scatter(x=slot_ricors, y=bound_upper, mode='lines', line_color='grey', showlegend=False))
        fig.add_trace(go.Scatter(x=slot_ricors, y=bound_lower, fill='tonexty', mode='lines', name='Error', line_color='grey'))

        fig.update_layout(xaxis_title="Day from 22.02", yaxis_title="INFECTED")        
        st.plotly_chart(fig)
    else:
        st.error('If the society does not take due'
                 ' precautions with social isolation, the Exponential trend can only diverge')


if __name__ == '__main__':
    main()
