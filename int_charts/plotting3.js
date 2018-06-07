trace1 = {
  x: [0.0, 1.0],
  y: [0.0, 1.0],
  line: {
    color: 'rgb(205, 12, 24)',
    dash: 'dash',
    width: 4
  },
  name: 'Luck',
  type: 'scatter',

};
trace2 = {
  x: [0., 0., 1.],
  y: [0.2, 1., 1.],
  line: {
    width: 2
  },
  name: 'ROC fold 1 (AUC = 1.00)',
  type: 'scatter',

};
trace3 = {
  x: [0., 0., 1.],
  y: [0.2, 1., 1.],
  line: {
    width: 2
  },
  name: 'Luck',
  type: 'scatter',

};
trace4 = {
  x: [0., 0., 1.],
  y: [0.2, 1., 1.],
  line: {
    width: 2
  },
  name: 'Luck',
  type: 'scatter',

};
trace5 = {
  x: [0., 0., 1.],
  y: [0.2, 1., 1.],
  line: {
    width: 2
  },
  name: 'Luck',
  type: 'scatter',

};
trace6 = {
  x: [0., 0., 1.],
  y: [0.2, 1., 1.],
  line: {
    width: 2
  },
  name: 'Luck',
  type: 'scatter',
};
trace11 = {
  x: [0.0, 0.010101010101010102, 0.020202020202020204, 0.030303030303030304, 0.04040404040404041, 0.05050505050505051, 0.06060606060606061, 0.07070707070707072, 0.08080808080808081, 0.09090909090909091, 0.10101010101010102, 0.11111111111111112, 0.12121212121212122, 0.13131313131313133, 0.14141414141414144, 0.15151515151515152, 0.16161616161616163, 0.17171717171717174, 0.18181818181818182, 0.19191919191919193, 0.20202020202020204, 0.21212121212121213, 0.22222222222222224, 0.23232323232323235, 0.24242424242424243, 0.25252525252525254, 0.26262626262626265, 0.27272727272727276, 0.2828282828282829, 0.29292929292929293, 0.30303030303030304, 0.31313131313131315, 0.32323232323232326, 0.33333333333333337, 0.3434343434343435, 0.3535353535353536, 0.36363636363636365, 0.37373737373737376, 0.38383838383838387, 0.393939393939394, 0.4040404040404041, 0.4141414141414142, 0.42424242424242425, 0.43434343434343436, 0.4444444444444445, 0.4545454545454546, 0.4646464646464647, 0.4747474747474748, 0.48484848484848486, 0.494949494949495, 0.5050505050505051, 0.5151515151515152, 0.5252525252525253, 0.5353535353535354, 0.5454545454545455, 0.5555555555555556, 0.5656565656565657, 0.5757575757575758, 0.5858585858585859, 0.595959595959596, 0.6060606060606061, 0.6161616161616162, 0.6262626262626263, 0.6363636363636365, 0.6464646464646465, 0.6565656565656566, 0.6666666666666667, 0.6767676767676768, 0.686868686868687, 0.696969696969697, 0.7070707070707072, 0.7171717171717172, 0.7272727272727273, 0.7373737373737375, 0.7474747474747475, 0.7575757575757577, 0.7676767676767677, 0.7777777777777778, 0.787878787878788, 0.797979797979798, 0.8080808080808082, 0.8181818181818182, 0.8282828282828284, 0.8383838383838385, 0.8484848484848485, 0.8585858585858587, 0.8686868686868687, 0.8787878787878789, 0.888888888888889, 0.8989898989898991, 0.9090909090909092, 0.9191919191919192, 0.9292929292929294, 0.9393939393939394, 0.9494949494949496, 0.9595959595959597, 0.9696969696969697, 0.9797979797979799, 0.98989898989899, 1.0],
  y: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  line: {
    width: 4
  },
  color: 'rgb(24, 12, 205)',
  name: 'Mean (AUC = 0.99)',
  type: 'scatter',
};

data = [trace1, trace2, trace3, trace4, trace5, trace6,trace11];
layout = {
title: 'A sample ROC curve',
xaxis: {title: 'False Positive Rate'},
yaxis: {title: 'True Positive Rate'}
};
Plotly.plot('plotly-div-3', {
  data: data
});