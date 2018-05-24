TEST = document.getElementById('pytestplot');

var trace1 = {
	x: [53, 54, 55, 56],
	y: [63, 64, 65, 66],
	name: '',
	line:{
		width: 0,
	},
	fill: 'none',
	mode: 'lines',
	type: 'scatter'
};

var trace2 = {
	x: [43, 44, 45, 46],
	y: [63, 64, 65, 66],
	name: 'Mean ±1 standard deviation',
	line:{
		width: 0,
	},
	fill:'tonexty',
	mode: 'lines',
	type: 'scatter'
};

var trace3 = {
	x: [0, 1],
	y: [0, 1],
	name: 'Luck',
	line: {
		dash: 'dot',
		color: 'rgb(255, 0, 0)',
	},
	mode: 'lines',
	type: 'scatter'
};

var trace4 = {
	x: [1, 2, 3, 4],
	y: [9, 8, 7, 6],
	name: 'ROC fold 4',
	line:{
		width: 2,
	},
	type: 'scatter'
};

var trace5 = {
	x: [11, 12, 13, 14],
	y: [19, 18, 17, 16],
	name: 'ROC fold 5',
	line:{
		width: 2,
	},
	type: 'scatter'
};

var trace6 = {
	x: [33, 44, 55, 66],
	y: [99, 98, 97, 96],
	name: 'Mean',
	line: {
		color: 'rgb(0, 0, 225)',
		width: 8,
	},
	mode: 'lines',
	type: 'scatter'
};
var data = [trace1, trace2, trace3, trace4, trace5, trace6];

Plotly.newPlot(TEST, data);