from pylab import *

delta = 0.01
x = arange(-3.0, 3.0, delta)
y = arange(-3.0, 3.0, delta)
X,Y = meshgrid(x, y)
Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
Z = Z2 - Z1 # difference of Gaussians

cmap = cm.get_cmap('rainbow', 10)    # PiYG
cmap_colors = cmap._segmentdata

def print_hex(r,b,g):
               if not(0 <= r <= 255 or 0 <= b <= 255 or 0 <= g <= 255):
                              raise ValueError('rgb not in range(256)')
               print('#%02x%02x%02x' % (r, b, g))


for i in range(len(cmap_colors['blue'])):
               r = int(cmap_colors['red'][i][2]*255)
               b = int(cmap_colors['blue'][i][2]*255)
               g = int(cmap_colors['green'][i][2]*255)
               print_hex(r, g, b)



im = imshow(Z, cmap=cmap, interpolation='bilinear',
            vmax=abs(Z).max(), vmin=-abs(Z).max())
axis('off')
colorbar()

show()
