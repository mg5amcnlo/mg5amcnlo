from __future__ import division
import collections
import math
import os

try:
    import madgraph
except ImportError:
    import internal.sum_html as sum_html
else:
    import madgraph.madevent.sum_html as sum_html

class grid_information(object):

    start, stop = -1,1 #original interval


    def __init__(self):
        # information that we need to get to create a new grid
        self.grid_base = collections.defaultdict(int)
        self.original_grid = collections.defaultdict(int) 
        self.non_zero_grid = collections.defaultdict(int)
        self.ng =0
        self.maxinvar=0
        self.nonzero = 0
        self.max_on_axis = collections.defaultdict(lambda: -1)
        self.min_on_axis = collections.defaultdict(lambda: 1)
        # information that we need to evaluate the cross-section/error
        self.sum_wgt = 0
        self.sum_abs_wgt = 0
        self.sum_wgt_square =0
        self.max_wgt = 0
        self.nb_ps_point = 0
        self.target_evt = 0
        
        #
        self.results = sum_html.Combine_results('combined')
        

    def convert_to_number(self, value):
        return float(value.replace('d','e'))

    def add_one_grid_information(self, path):

        if isinstance(path, str):
            finput = open(path)
        elif isinstance(path, file):
            finput=path
        else:
            raise Exception, "path should be a path or a file descriptor"
        
        line = finput.readline()
        if self.nonzero == 0:
            #first information added
            self.nonzero, self.ng, self.maxinvar = [int(i) for i in line.split()]
            
        else:
            nonzero, ng, maxinvar = [self.convert_to_number(i) for i in line.split()]
            self.nonzero+=nonzero
            assert ng == self.ng
            assert maxinvar == self.maxinvar


        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        for j in range(self.maxinvar):
            for i in range(self.ng): 
                self.grid_base[(i,j)] = data.pop(0)

        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        for j in range(self.maxinvar):
            for i in range(self.ng): 
                self.original_grid[(i,j)] = data.pop(0)

        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        for j in range(self.maxinvar):
            for i in range(self.ng): 
                self.non_zero_grid[(i,j)] = int(data.pop(0))

        #minimal value for each variable of integraion
        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        for j in range(self.maxinvar):                
            self.min_on_axis[j] = min(self.min_on_axis[j],data.pop(0))

        #maximum value for each variable of integraion
        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        for j in range(self.maxinvar):                
            self.max_on_axis[j] = max(self.max_on_axis[j],data.pop(0))

        # cumulative variable for the cross-section
        line = finput.readline()
        data = [self.convert_to_number(i) for i in line.split()]
        self.sum_wgt += data[0]
        self.sum_abs_wgt += data[1]
        self.sum_wgt_square += data[2]
        self.max_wgt = max(self.max_wgt, data[3])
        self.nb_ps_point += data[4]
        if self.target_evt:
            assert self.target_evt == data[5], "%s != %s" % (self.target_evt, data[5])
        else: 
            self.target_evt += data[5]  
        
        
    def add_results_information(self, path):
        
        if isinstance(path, str):
            finput = open(path)
            fname = path
        elif isinstance(path, file):
            finput=path
            fname = finput.name
        else:
            raise Exception, "path should be a path or a file descriptor"
         
        
        self.results.add_results(fname,finput)

    
    def write_associate_grid(self, path):
        """use the grid information to create the grid associate"""

        new_grid = self.get_new_grid()
        
        fsock = open(path, 'w')
        data = []
        for var in range(self.maxinvar):
            for i in range(self.ng):
                data.append(new_grid[(i,var)])
        
        while len(data) >= 4:
            v1, v2, v3, v4 = data[:4]
            data = data[4:]
            fsock.write('%+.16f %+.16f %+.16f %+.16f \n' % (v1, v2, v3, v4))
        
        # if data is not a multiple of 4 write the rest.
        for v in data:
            fsock.write('%+.16f' % v)
        if  data:
            fsock.write('\n')
            
            
    def get_cross_section(self):
        """return the cross-section error"""
        
        mean = self.sum_wgt*self.target_evt/self.nb_ps_point
        rmean =  self.sum_abs_wgt*self.target_evt/self.nb_ps_point
        
        vol = 1/self.target_evt
        sigma = self.sum_wgt_square/vol**2
        sigma -= self.nonzero * mean**2
        sigma /= self.nb_ps_point*(self.nb_ps_point -1)

        #print 'integral', mean
        #print 'integral of the absolute fct', rmean        
        #print 'error', math.sqrt(abs(sigma))
        #print 'info', self.sum_wgt_square, vol, self.nonzero, mean, 1, self.nb_ps_point
        
        return mean, rmean, math.sqrt(abs(sigma))
        
        
            


    def get_new_grid(self):
        
        new_grid = collections.defaultdict(int)
        
        for var in range(self.maxinvar):
            one_grid = self.get_new_grid_for_var(var)
            for j,value in enumerate(one_grid):
                new_grid[(j,var)] = value
        
        return new_grid
        


    def get_new_grid_for_var(self, var):
        
        #1. biais the grid to allow more points where the fct is zero.    
        grid = collections.defaultdict(int)
        for i in range(self.ng):
            if self.non_zero_grid[(i, var)] != 0:
                factor = min(10000, self.nonzero/self.non_zero_grid[(i,var)])
                grid[(i, var)] = self.grid_base[(i, var)] * factor


        #2. average the grid
        def average(a,b,c):
            if b==0:
                return 0
            elif a==0:
                return (b+c)/2
            elif c==0:
                return (a+b)/2
            else:
                return (a+b+c)/3
        
        tmp_grid = collections.defaultdict(int)
        for j in range(self.ng):
            tmp_grid[(j, var)] = average(grid[(j-1, var)],grid[(j, var)],grid[(j+1, var)])
        grid = tmp_grid


                                    
        #3. takes the logs to help the re-binning to converge faster
        sum_var = sum([grid[(j,var)] for j in range(self.ng)])  
        for j in range(self.ng):
            if grid[(j,var)]:
                x0 = 1e-14+grid[(j,var)]/sum_var
                grid[(j,var)] = ((x0-1)/math.log(x0))**1.5
        

        start, stop = 0, self.ng-1
        start_bin, end_bin = 0, 1 
        test_grid = [0]*self.ng   # a first attempt for the new grid

        # special Dealing with first/last bin for handling endpoint.
        xmin, xmax = self.min_on_axis[var], self.max_on_axis[var]
        if (xmin- (-1) > (self.original_grid[(1,var)] - self.original_grid[(0,var)])):
            start = 1
            start_bin = xmin - (self.original_grid[(1,var)] - self.original_grid[(0,var)])/5
            test_grid[0] = start_bin
        else:
            xmin = -1
        if (1- xmax) > (self.original_grid[(self.ng-1,var)] - self.original_grid[(self.ng-2, var)]):
            stop = self.ng -2
            xmax = xmax + (self.original_grid[(self.ng-1,var)] - self.original_grid[(self.ng-2, var)])/5
            test_grid[self.ng-1] = xmax
        else:
            xmax = 1
            test_grid[self.ng-1] = xmax


        #compute the value in order to have the same amount in each bin
        sum_var = sum([grid[(j,var)] for j in range(self.ng)])  
        avg = sum_var / (stop-start+1)
        cumulative = 0
        pos_in_original_grid = -1
        for j in range(start,stop):
            while cumulative < avg and pos_in_original_grid < self.ng:
                #the previous bin (if any) is fully belonging to one single bin
                #of the new grid. adding one to cumulative up to the point that 
                #we need to split it
                pos_in_original_grid += 1
                cumulative += grid[(pos_in_original_grid, var)]
                start_bin = end_bin
                end_bin = max(xmin,min(xmax,self.original_grid[(pos_in_original_grid, var)]))
            cumulative -= avg
            #if pos_in_original_grid == 0:
            #    print grid[(pos_in_original_grid,var)]
            #    print cumulative
            if end_bin != start_bin and cumulative and grid[(pos_in_original_grid,var)]: 
                test_grid[j] = end_bin - (end_bin-start_bin)*cumulative / \
                                                grid[(pos_in_original_grid,var)]
            else:
                test_grid[j] = end_bin
                      
        # Ensure a minimal distance between each element of the grid
        sanity = True
        for j in range(1, self.ng):
            if test_grid[j] - test_grid[j-1] < 1e-14:
                test_grid[j] = test_grid[j-1] + 1e-14
            if test_grid[j] > xmax:
                sanity = False
                break
        # not in fortran double check of the sanity from the top.
        if not sanity:
            for j in range(1, self.ng):
                if test_grid[-1*j] > xmax - j * 1e-14:
                    test_grid[-1*j] = xmax - j * 1e-14
                
        # return the new grid
        return test_grid


if __name__ == "__main__":
    o = grid_information()
    o.add_one_grid_information('./grid_information')
    o.write_associate_grid('./mine')
            
        

        
        
                
                
                

            
            
            
            
                          
        
        
        
        
        
        
        


            
            


