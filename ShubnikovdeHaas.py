import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.special import jv
import re
from scipy.ndimage import gaussian_filter1d
from os import listdir
from os.path import isfile, join
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
#
#Author: Denis Aglagul
#Affiliation: RPI physics
#


class samples:
    '''calibration:[a,b] such that x_cal = (x_old + a)*b.
        Examples: cornell [a,b] = [-5,2.8] and for RPI [a,b] = [0,10]
        
        Container: A string of the folder path
            Example:'/home/mourning-dove/Data/Shubnikov_de_Haas/1B' which points to data
            for a sample named 1B
            
        save: A 2 index list describting how the data is saved.
            example: [NCSU_names,'/t'] where NCSU_names = ['t','x','y','x2',y2']
             and '/t' denotes a tab delimiter
        blur: gaussian convolution, set to 1 for no change.
        
        Holder: describes how the sample is mounted
        examples: Universal Horizontal, Universal Vertical, Fin, etc...'''
        
    def __init__(self,Container,save,calibration,blur,Holder):

        self.Container = Container
        self.save = save
        self.calibration = calibration 
        self.blur = blur
        self.Holder = Holder
        self.raw_data = {}
        self.invertedB = {}
        self.FFT = {}


    def load_data(self,range,**kwargs):

        """Load all data form a folder. This method does not 'dig' into folders.
        Data is then calibrated using the given calibration [a,b] 
        if parameter is not given: this returns a list of dataframes.
        if given: returns a dictionary where the parameters are the keys and the dataframes are the items
        
        optional arguement:
        driection = +1 or -1 to choosesweep direciton
        parameter: in each filename, if you have saved a parameter such as deg for angles, K for temp or mA for current etc
        set parameter equal to this unit as a string.
        examples, if you have data saved as 5mA_0deg, 5mA_7.5deg, 6mA_9deg and want to interogate the angle dependence, then
        set parameter='deg' which will then make your output a dictionary where the keys are '0','7.5', and '9'.
        Current: if you saved data as a voltage and the current sourced is saved in the file names as (...)mA then setting this to true will
        convert to a resistance.
        """
        self.range = range
        direction = kwargs.get('direction',None)
        parameter = kwargs.get('parameter',None) #might be valuable to make a dictionary of dictionarys for several paremeters 
        Grad = kwargs.get('Grad',False)

        Current = kwargs.get('Current',False)
        Sample_set = []
        self.parameter_list = []
        

        onlyfiles = [f for f in listdir(self.Container) if isfile(join(self.Container, f))]
        
    


        for set in onlyfiles:
            try:
                if self.Container.endswith('Parameters') == True:
                    continue
                Data = pd.read_csv(self.Container + '/'+ set,header=None, delimiter=self.save[1], names =  self.save[0])
                if direction != None:
                    temp_mask  = (gaussian_filter1d(direction*np.diff(Data.t),50) > 0.01)
                    Data =  Data[0:len(np.diff(Data.t))]                     
                    Data = Data[temp_mask]

                Data.x = gaussian_filter1d(Data.x,self.blur)
               

                Data.x = Data.x
                Data.t = gaussian_filter1d(Data.t,self.blur)
                Data.t = (Data.t - self.calibration[0]) * self.calibration[1]


                mask = (Data.t >= min(range)) & (Data.t <= max(range))
                
                
            except:
                
                continue
            if Grad:
                    try:
                        Data.x = np.gradient(np.gradient(Data.x))
                    except ValueError:
                        continue
            Data = Data[mask]

            if Current:    
                match = re.search('(\d+)mA', set) 
                pmatch =re.search('(\d+)p', set)
                if pmatch:
                    current = int((pmatch.group(1)))+ int((match.group(1)))/(10**np.ceil(np.log10(int((match.group(1)))))) # the number the the right of 'p' is meant to be a decimal
                elif match:                                                                                            # but I read it a int(i.e .375 -> 375) so ceil(log375) =3
                    current  = int((match.group(1)))                                                                       # and 10^3 = 1000 thus so 375/(10**ceillog375) = 375/10**3 = 0.375 
                else:
                    current = 1
                
                Data.x = Data.x/current
            
                
            match = re.search('(\d+)'+parameter, set) 
            pmatch =re.search('(\d+)p', set)
            negmatch = re.search('neg(\d+)',set)
            
            if pmatch:
                par = int((pmatch.group(1)))+ int((match.group(1)))/(10**np.ceil(np.log10(int((match.group(1)))))) # the number the the right of 'p' is meant to be a decimal
            elif match:                                                                                            # but I read it a int(i.e .375 -> 375) so ceil(log375) =3
                par  = int((match.group(1)))       
            elif negmatch:
                par  = -int((match.group(1)))                                                                # and 10^3 = 1000 thus so 375/(10**ceillog375) = 375/10**3 = 0.375 
            else:
                par = 0
            self.parameter_list.append(par)
            Sample_set.append(Data)
        

        if parameter == None:
            self.raw_data = Sample_set
            return  Sample_set
        else:
            parameter_dict = {angle:sweep for angle,sweep in zip(self.parameter_list,Sample_set)}   
              #output results as a dictionary where the keys are the parameters
            self.raw_data = dict(sorted(parameter_dict.items(), key=lambda item: item[0]))
            return parameter_dict
        


    def invertB(self,**kwargs):

        '''transforms somewhat messy data thats linear in B to 1/B
        overunder: if you want to OVERsample or UNDERsample, leave empty to do neither( default to 1) to over sample 2x :
        'overunder' = 2, and to undersample by 2: 'overunder' = 0.5 this will round for your.'''
        #TODO handle inversion near 0 field

        overunder = kwargs.get('overunder',1)
        loner = kwargs.get('loner',False)

        polishedV = []
        data = self.raw_data
        


        if loner: #if we are using a single data set
            crossX = data.t
            crossY = data.x
            al = np.dstack((np.asarray(crossX),crossY))
            al =al[0,:,:]                       
            df = pd.DataFrame(al,columns = ['t','v'])                   ####1
            L1 =df.groupby(['t'],as_index = False,sort = True).mean()  
            
            xdata_for_spline = L1.t
            xdata_for_spline = xdata_for_spline.to_numpy()
            ydata = L1.v
            ydata = ydata.to_numpy()
            try:
                cs = CubicSpline(xdata_for_spline,ydata)
                xs = np.arange(min(xdata_for_spline),max(xdata_for_spline), 1/(int(overunder*len(crossX))))   
                polishedV=cs(xs) #record background subtracted curves   
                out = np.asanyarray(polishedV)    
                self.invertedB = out
            
                return out
            except ValueError:
                return []
        
        elif not loner: # if we are considering a parameter sweep.
            new_dicts = {}
            for key,value in data.items(): 
                crossY = value.x
                crossX = value.t**-1
                
                al = np.dstack((np.asarray(crossX),crossY))
                al =al[0,:,:]       
                df = pd.DataFrame(al,columns = ['t','v'])                   ####1
                L1 =df.groupby(['t'],as_index = False,sort = True).mean()  

                polishedV = []
                xdata_for_spline = L1.t
                xdata_for_spline = xdata_for_spline.to_numpy()
                ydata = L1.v
                ydata = ydata.to_numpy()
                
                try: #try to arrange a 1/B data set, if the set is empty this fails and raises the excpetion
                    
                    xs = np.arange(min(xdata_for_spline),max(xdata_for_spline), 1/((overunder*len(crossX))))
                    cs = CubicSpline(xdata_for_spline,ydata)
                    xscs= np.dstack((np.asarray(xs),cs(xs)))       
                    polishedV= pd.DataFrame(xscs[0,:,:],columns = ['t','x']) #record background subtracted curves and package everything into data frame

                except ValueError: # skip the empty data set
                    
                    continue

                new_dicts[key] = polishedV
  


            self.invertedB = new_dicts
            return new_dicts
        

    def FourierTransform(self,**kwargs):
        '''Outputs the fourier transform of what is stored in  whatever...
        the output will shift the frequencies but does not take do anything to the FFT amplitudes. take the abs() to
        get a familiar result'''
        load = kwargs.get('load',self.invertedB)
        
        FFT_dict = {}
        
        for k,d in load.items():
           
            Resistance = d.x
            N =100 * len(d.t) 
            
            fourierTransform = np.fft.fft(Resistance,n=N)/len(Resistance)           # Normalize amplitude
            fourierTransform = fourierTransform[ range(int(N/2) )  ]       # Exclude sampling frequency 
            fourierTransform /= np.max(fourierTransform)
            freqs = np.fft.fftfreq(N)  
            freqs = np.abs(freqs[ range(int(N/2) )  ]  / (np.mean(np.diff(d.t))) ) 
            FF = np.dstack((np.asarray(freqs),fourierTransform))
            FFT_dict[k] = pd.DataFrame(FF[0,:,:],columns = ['freqs','amp']) #record background subtracted curves and package everything into data frame

        self.FFT = FFT_dict
        return self.FFT
    

    def stfft(self,**kwargs):
        load = kwargs.get('load',self.invertedB)
        window =kwargs.get('window',gaussian(
            10,
              3,
              sym = True))
        print(int(np.mean([len(d.t) for k,d in self.invertedB.items()])))
        sxd = {}
        for k,d in load.items():
            print(d.x)
            #TODO I should use the same window for all of them
            try:
                SFT = ShortTimeFFT(window,
                                hop = 30,
                                mfft = 5000,
                                fs= 1/np.mean(np.abs(np.diff(d.t))),
                                scale_to = 'magnitude')
                sx  = SFT.stft(np.asarray(d.x))
                

                sxd[k] = sx
            except:
                continue
        return sxd


        


if __name__ == '__main__':
    pass