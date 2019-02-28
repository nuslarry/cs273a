from matplotlib import pyplot

a = [ pow(2,i) for i in range(10) ]

pyplot.subplot(2,1,1)
print(arange(0,10,1))
pyplot.plot(a, arange(0,10,1) , color='blue', lw=2)
pyplot.xscale('log',basex=2)
pyplot.show()