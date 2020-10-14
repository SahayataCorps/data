import os
import numpy
import matplotlib
from matplotlib import pyplot, image

FONT = {	'family': 'Keyboard',
		'size': 20}
matplotlib.rc('font', **FONT)

#This function creates mathplotlib figure and axis, then assign poll image to it, and then return them to the calling function
def drawDisplayImage(dispsize, imagefile=None):
	_, ext = os.path.splitext(imagefile)
	ext = ext.lower()
	data_type = 'float32' if ext == '.png' else 'uint8'
	screen = numpy.zeros((dispsize[1],dispsize[0],3), dtype=data_type)
	if imagefile != None:
		if not os.path.isfile(imagefile):
			raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
		img = image.imread(imagefile)
		if not os.name == 'nt':
			img = numpy.flipud(img)
		w, h = len(img[0]), len(img)
		x = dispsize[0]//2 - w//2
		y = dispsize[1]//2 - h//2
		screen[y:int(y+h),x:int(x+w),:] += img
	dpi = 100.0
	figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
	fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
	ax = pyplot.Axes(fig, [0,0,1,1])
	ax.set_axis_off()
	fig.add_axes(ax)
	ax.axis([0,dispsize[0],0,dispsize[1]])
	ax.imshow(screen)#, origin='upper')	
	return fig, ax

#This function is to change data into dictionary and remove any unwanted data
def singleParseFixations(fixations):
	fix = {	'x':numpy.zeros(len(fixations)), 'y':numpy.zeros(len(fixations)), 'dur':numpy.zeros(len(fixations))}
	for fixnr in range(len(fixations)):
		stime, etime, dur, ex, ey = fixations[fixnr]
		fix['x'][fixnr] = ex
		fix['y'][fixnr] = ey
		fix['dur'][fixnr] = dur
	return fix

#Used by heatmap
def gaussian(x, sx, y=None, sy=None):
	if y == None:
		y = x
	if sy == None:
		sy = sx
	# centers	
	xo = x/2
	yo = y/2
	# matrix of zeros
	M = numpy.zeros([y,x],dtype=float)
	# gaussian matrix
	for i in range(x):
		for j in range(y):
			M[j,i] = numpy.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )

	return M

#This function plot fixation by plotting scatter plots and assigning the duration of point as weight 
def plotFixation(fix, dispsize, imagefile, alpha=0.5, savefilename=None):
	fig, ax = drawDisplayImage(dispsize, imagefile)
	siz = 1 * (fix['dur']*20)
	col = '#73d216'
	ax.scatter(fix['x'],fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='none')
	ax.invert_yaxis()
	if savefilename != None:
		fig.savefig(savefilename)
	else:
		return (fig, ax)

#This function plot heatmap
def plotHeatMap(fix, dispsize, imagefile ,durationweight=True, alpha=0.5, savefilename=None):
    fig, ax = drawDisplayImage(dispsize, imagefile)
    gwh = 200
    gsdwh = gwh//6
    gaus = gaussian(gwh,gsdwh)
	# matrix of zeroes
    strt = gwh//2
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)

    heatmap = numpy.zeros(heatmapsize)
	# create heatmap
    for i in range(0,len(fix['dur'])):
        # get x and y coordinates
        #x and y - indexes of heatmap array. must be integers
        x = strt + int(fix['x'][i]) - int(gwh/2)
        y = strt + int(fix['y'][i]) - int(gwh/2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj=[0,gwh];vadj=[0,gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:
            heatmap[y:y+gwh,x:x+gwh] += gaus * fix['dur'][i]
	# resize heatmap
    heatmap = heatmap[strt:dispsize[1]+strt,strt:dispsize[0]+strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap>0])
    heatmap[heatmap<lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)
    else:
        return (fig, ax)

#This function plot scanpath with both fixation or heatmap (as given by user)
def plotScanpath(fix, saccades,dispsize, imagefile=None, alpha=0.5, savefilename=None, withFixation = True, withHeatmap = False, singleUser= True):
    
    if withFixation:
        fig , ax = drawDisplayImage(dispsize, imagefile)
        ax.scatter(fix['x'],fix['y'], s=(1 * fix['dur']*30 ), c='#4e9a06', marker='o', cmap='jet', alpha=alpha, edgecolors='none')	

    if withHeatmap:
        fig, ax = plotHeatMap(fix, dispsize, imagefile ,durationweight=True, alpha=0.5)
        

	# draw annotations (fixation numbers)
    for i in range(len(fix['x'])):
        ax.annotate(str(i+1), (fix['x'][i],fix['y'][i]), color='#2e3436', alpha=1, horizontalalignment='center', verticalalignment='center', multialignment='center')

	# SACCADES
    if saccades:
        if singleUser:
            for st, et, dur, sx, sy, ex, ey in saccades:
                ax.arrow(sx, sy, ex-sx, ey-sy, alpha=alpha, fc='#eeeeec', ec='#2e3436', fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)
        else:
            for sx, sy, cost, dur, nop, ex, ey in saccades:
                ax.arrow(sx, sy, ex-sx, ey-sy, alpha=alpha, fc='#eeeeec', ec='#2e3436', fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

    if not withHeatmap:
        ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig

