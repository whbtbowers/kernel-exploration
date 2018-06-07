SCATTER = document.getElementById('scatter');
var trace1 = {
	x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	y: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
	marker: {
		size: 6,
		opacity: 0.5,
		symbol: 'square',
	},
	mode: 'markers',
	name: 'Diabetes',
	type: 'scatter',
};
var trace2 = {
	x: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
	y: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
	marker: {
		size: 6,
		opacity: 0.5,
		symbol: 'square',
	},
	mode: 'markers',
	name: 'Non-diabetes',
	type: 'scatter',
};
var data = [trace1, trace2];

Plotly.plot(SCATTER, data);