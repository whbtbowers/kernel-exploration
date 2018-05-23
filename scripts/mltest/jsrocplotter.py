import p2funcs as p2f

def js_fold_line(x_vals_list, y_vals_list):

    #Declare trace list for calling on data
    trace_list = []
    traces = []

    for i in range(len(x_vals_list)):

        trace = 'trace%s' % str(i+4)
        trace_list.append(trace)
        open_trace = 'var %s = {' % trace
        close_trace = "\n\ttype: 'scatter'\n};\n"
        trace_x = '\n\tx: ' + str(x_vals_list[i]) + ','
        trace_y = '\n\ty: ' + str(y_vals_list[i]) + ','
        leg_name = "\n\tname: 'ROC fold %s," % str(i+4)
        line = "\n\tline:{\n\t\twidth: 2,\n\t},"
        full_trace = open_trace + trace_x + trace_y +  leg_name + line + close_trace
        traces.append(full_trace)

    return(trace_list, traces)

def js_mean_trace(mean_x, mean_y, trace_list, traces):

    mean_trace = "trace" + str(int(trace_list[-1][5:]) + 1)
    trace_list.append(mean_trace)
    mean_trace_x = '\n\tx: ' + str(mean_x) + ','
    mean_trace_y = '\n\ty: ' + str(mean_y) + ','
    mean_trace_close ="\n\tname: 'Mean',\n\tline: {\n\t\tcolor: 'rgb(0, 0, 225)',\n\t\twidth: 8,\n\t},\n\tmode: 'lines',\n\ttype: 'scatter'\n};"
    mean_trace_full = "var %s = {%s%s%s" % (mean_trace, mean_trace_x, mean_trace_y, mean_trace_close)
    traces.append(mean_trace_full)

    return(trace_list, traces)

def construct_js_roc(chartname, divname, trace_list, traces):

    print("%s = document.getElementById('%s')\n" % (chartname, divname))
    for trace in traces:
        print(trace)
    print('\nvar data = %s;' % str(trace_list))
    print('\nPlotly.newPlot(%s, data)' % chartname)

def js_luck_trace(trace_list, traces):

    trace_list = ['trace3'] + trace_list
    traces = ["var trace3 = {\n\tx: [0, 1],\n\ty: [0, 1],\n\tname: 'Luck',\n\tline: {\n\t\tdash: 'dot',\n\t\tcolor: 'rgb(255, 0, 0)',\n\t},\n\tmode: 'lines',\n\ttype: 'scatter'\n};"] + traces
    return(trace_list, traces)

def js_tpr_std(tpr_std_upper, tpr_std_lower, fpr_std, trace_list, traces):

    trace_list = ['trace2'] + trace_list
    traces = ["var trace2 = {\n\tx: %s,\n\ty: %s\n\tname: 'Mean ±1 standard deviation',\n\tline:{\n\t\twidth: 0,\n\t},\n\tfill:'tonexty',\n\tmode: 'lines',\n\ttype: 'scatter'\n};\n" % (str(tpr_std_upper), str(fpr_std))] + traces

    trace_list = ['trace1'] + trace_list
    traces = ["var trace1 = {\n\tx: %s,\n\ty: %s,\n\tname: '',\n\tline:{\n\t\twidth: 0,\n\t},\n\tfill: 'none',\n\tmode: 'lines',\n\ttype: 'scatter'\n};\n" % (str(tpr_std_lower), str(fpr_std))] + traces
    return(trace_list, traces)


bs = [[1, 2, 3, 4],[11, 12, 13, 14]]
cs = [[9, 8, 7, 6], [19, 18, 17, 16]]
mean_x = [33, 44, 55, 66]
mean_y = [99, 98, 97, 96]
up_lims = [43, 44, 45, 46]
low_lims = [53, 54, 55, 56]
fprs = [63, 64, 65, 66]

trace_list, traces = js_fold_line(bs, cs)
trace_list, traces = js_mean_trace(mean_x, mean_y, trace_list, traces)
trace_list, traces = js_luck_trace(trace_list, traces)
trace_list, traces = js_tpr_std(up_lims, low_lims, fprs, trace_list, traces)

construct_js_roc('TEST', 'pytestplot', trace_list, traces)
