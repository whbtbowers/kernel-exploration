heatMap = document.getElementById('funcheat');

var trace1 = {
	x: ['Linear KPCA', 'RBF KPCA', 'Laplacian KPCA', ' Cosine KPCA', 'Sigmoid KPCA'],
	y: ['Linear SVM', 'RBF SVM', 'Sigmoid SVM'],
	z: [[0.8, 2.4, 2.5, 3.9, 0.0], [2.4, 0.0, 4.0, 1.0, 2.7], [1.1, 2.4, 0.8, 4.3, 1.9]],
	colorscale: 'YIOrRd',
	type: 'heatmap',
	colorbar:{
		title:'Mean area under ROC curve',
		titlefont: {color: '#ffffff'},
		titleside:'right',
		tickfont: {
			color: '#ffffff'
		},

	},
};

var data = [trace1];

var layout = {
	paper_bgcolor:'#2e3141',
	plot_bgcolor: '#2e3141',
	legend: {
		bgcolor: '#FFFFFF',
		font: {
			color: '#ffffff',
		},
	},
	paper_bgcolor:'#2e3141',
	plot_bgcolor: '#2e3141',
	xaxis1: {
		color: '#ffffff',
		gridcolor: '#E1E5ED',
		tickfont: {color: '#ffffff'},
		title: '',
		titlefont: {color: '#ffffff'},
		zerolinecolor: '#E1E5ED'
	},
	yaxis1: {
		color: '#ffffff',
		gridcolor: '#E1E5ED',
		tickfont: {color: '#ffffff'},
		title: '',
		titlefont: {color: '#4D5663'},
		zeroline: false,
		zerolinecolor: '#E1E5ED'
	}
};

Plotly.plot(heatMap, data, layout);

heatMap.on('plotly_click', function(data){
	var xcoord = x.indexOf(data.points.x);
	var ycoord = y.indexOf(data.points.y);
	console.log(xcoord, ycoord);
	//var corr = strList[char];
	//console.log(corr + '_roc.html');
	//window.open('testplot.html');
});
