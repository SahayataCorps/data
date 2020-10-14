"""
This is a test file altered to create graph on videos
"""
# native
import os
# external
import numpy
import matplotlib
from matplotlib import pyplot, image
import cv2

# # # # #
# LOOK

# COLOURS
# all colours are from the Tango colourmap, see:
# http://tango.freedesktop.org/Tango_Icon_Theme_Guidelines#Color_Palette
COLS = {	"butter": [	'#fce94f',
					'#edd400',
					'#c4a000'],
		"orange": [	'#fcaf3e',
					'#f57900',
					'#ce5c00'],
		"chocolate": [	'#e9b96e',
					'#c17d11',
					'#8f5902'],
		"chameleon": [	'#8ae234',
					'#73d216',
					'#4e9a06'],
		"skyblue": [	'#729fcf',
					'#3465a4',
					'#204a87'],
		"plum": 	[	'#ad7fa8',
					'#75507b',
					'#5c3566'],
		"scarletred":[	'#ef2929',
					'#cc0000',
					'#a40000'],
		"white":	[ '#ffffff',
					'#ffffff',
					'#ffffff'],
		"aluminium": [	'#eeeeec',
					'#d3d7cf',
					'#babdb6',
					'#888a85',
					'#555753',
					'#2e3436'],
		}
# FONT
FONT = {	'family': 'Keyboard',
		'size': 20}
matplotlib.rc('font', **FONT)


# # # # #
# FUNCTIONS

def draw_fixations(fixations, dispsize, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5, savefilename=None):


	# FIXATIONS
	fix = parse_fixations(fixations)

	# IMAGE
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# CIRCLES
	# duration weigths
	if durationsize:
		siz = 1 * (fix['dur']*10)

	else:
		siz = 1 * numpy.median(fix['dur']*10)
	if durationcolour:
		col = fix['dur']
	else:
		col = COLS['white'][2]
	# draw circles

	ax.scatter(fix['x'],fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='none')

	# FINISH PLOT
	# invert the y axis, as (0,0) is top left on a display
	ax.invert_yaxis()
	# save the figure if a file name was provided
	if savefilename != None:
		fig.savefig(savefilename)

	return fig


def draw_heatmap(fixations, dispsize, imagefile=None, durationweight=True, alpha=0.5, savefilename=None):


	# FIXATIONS
	fix = parse_fixations(fixations)

	# IMAGE
	fig, ax = draw_display(dispsize, imagefile=imagefile)

	# HEATMAP
	# Gaussian
	gwh = 200
	gsdwh = gwh/6
	gaus = gaussian(gwh,gsdwh)
	# matrix of zeroes
	strt = gwh/2
	heatmapsize = dispsize[1] + 2*strt, dispsize[0] + 2*strt
	heatmap = numpy.zeros(heatmapsize, dtype=float)
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
			# add Gaussian to the current heatmap
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

	return fig




def draw_scanpath(fixations, saccades,  imagefile=None, alpha=0.5, singleUser = True):
    fix = parse_fixations(fixations)

    frameNum = 0
    dpi = 100.0
    capt = cv2.VideoCapture(imagefile)
    dispsize = [int(capt.get(3)), int(capt.get(4))]
    figsize = (dispsize[0]//dpi, dispsize[1]//dpi)
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0,0,1,1])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.axis([0,dispsize[0],0,dispsize[1]])
    ax.invert_yaxis()
    i = 0
    durr = 0
    nextInd = 0


    while True:
        ret, img = capt.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if i==0:
                durr = fix["dur"][nextInd]
                i+=1
            else:
                i+=1
            print(frameNum)

            screen = numpy.zeros((dispsize[1], dispsize[0], 3))
            if not os.name == 'nt':
                img = numpy.flipud(img)

            w, h = len(img[0]), len(img)

            x = dispsize[0] // 2 - w // 2
            y = dispsize[1] // 2 - h // 2
            # draw the image on the screen
            screen[int(y):int(y + h), int(x):int(x + w), :] += img
            ax.imshow(screen.astype('uint8'))

            ax.scatter(fix['x'][nextInd], fix['y'][nextInd], s=(1 * fix['dur'][nextInd] * 30), c='#4e9a06', marker='o', cmap='jet', alpha=alpha,
                   edgecolors='none')

            ax.scatter(fix['x'][nextInd+1], fix['y'][nextInd+1], s=(1 * fix['dur'][nextInd+1] * 30), c='#4e9a06', marker='o', cmap='jet', alpha=alpha,
                   edgecolors='none')

            ax.annotate(str(nextInd + 1), (fix['x'][nextInd], fix['y'][nextInd]), color='#2e3436', alpha=1, horizontalalignment='center',
                            verticalalignment='center', multialignment='center')


            if singleUser:
                st, et, dur, sx, sy, ex, ey = saccades[nextInd]
                ax.arrow(sx, sy, ex - sx, ey - sy, alpha=alpha, fc='#eeeeec', ec='#2e3436', fill=True, shape='full',
                             width=10, head_width=20, head_starts_at_zero=False, overhang=0)
            else:
                sx, sy, cost, dur, nop, ex, ey = saccades[nextInd]
                ax.arrow(sx, sy, ex - sx, ey - sy, alpha=alpha, fc='#eeeeec', ec='#2e3436', fill=True, shape='full',
                             width=10, head_width=20, head_starts_at_zero=False, overhang=0)


            savefilename = "images/"+str(frameNum)+".jpg"
            fig.savefig(savefilename)
            frameNum+=1
            ax.clear()
            if i==durr:
                durr = 0
                i = 0
                nextInd+=1

        else:
            out = cv2.VideoWriter("_out.avi", cv2.VideoWriter_fourcc(*'XVID'),int(capt.get(5)),tuple(dispsize))
            for i in range(frameNum):
                frame = cv2.imread(str(i)+".jpg")
                out.write(frame)
            break



def gaussian(x, sx, y=None, sy=None):

	"""Returns an array of numpy arrays (a matrix) containing values between
	1 and 0 in a 2D Gaussian distribution

	arguments
	x		-- width in pixels
	sx		-- width standard deviation

	keyword argments
	y		-- height in pixels (default = x)
	sy		-- height standard deviation (default = sx)
	"""

	# square Gaussian if only x values are passed
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


def parse_fixations(fixations):

	"""Returns all relevant data from a list of fixation ending events

	arguments

	fixations		-	a list of fixation ending events from a single trial,
					as produced by edfreader.read_edf, e.g.
					edfdata[trialnr]['events']['Efix']

	returns

	fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
				a numpy array) for the x and y coordinates and duration of
				each fixation
	"""

	# empty arrays to contain fixation coordinates
	fix = {	'x':numpy.zeros(len(fixations)),
			'y':numpy.zeros(len(fixations)),
			'dur':numpy.zeros(len(fixations))}
	# get all fixation coordinates
	for fixnr in range(len( fixations)):
		stime, etime, dur, ex, ey = fixations[fixnr]
		fix['x'][fixnr] = ex
		fix['y'][fixnr] = ey
		fix['dur'][fixnr] = dur

	return fix
