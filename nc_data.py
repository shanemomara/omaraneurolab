# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 20:36:50 2017

@author: Raju
"""

import logging, inspect, collections
from collections import OrderedDict as oDict
import numpy as np
import numpy.random as nprand
import re
import nc_ext
from imp import reload
reload(nc_ext)
from nc_ext import CircStat, find, residualStat, extrema, butter_filter, fft_psd,\
             smooth1D, smooth2d, chopEdges, histogram, bhatt, hellinger,\
             histogram2d, linfit, find2d, rot2d, corrCoeff,findChunk
import scipy as sc
from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.signal as sg
from functools import reduce
from scipy.fftpack import fft
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import os
import h5py
import io

class Nhdf(object):
    def __init__(self, **kwargs):
        self._filename= kwargs.get('filename', '')
        self.f= None
        if os.path.exists(self._filename):
            self.File()

    def getFilename(self):
        return self._filename
    
    def setFilename(self, filename= None):
        if filename:
            self._filename= filename
        try:
            self.File()
        except:
            logging.error('Invalid file!')
    
    def getFileObject(self):
        if isinstance(self.f, io.IOBase):
            return self.f
        else:
            logging.warning('The file Nhdf instance is not open yet, use Nhdf.File() method to open it!')
            
    def File(self): # Named similar to h5py.File() to open the file and stire its object
        self.close()
        try:
            self.f= h5py.File(self._filename, 'a')
            self.initialize()
        except:
            logging.error('Cannot open '+ self._filename)   
        return self.f
    
    def close(self):
        if isinstance(self.f, h5py.File):
            self.f.close()
            self.f= None
            
    def initialize(self):
        groups= ['acquisition', 'processing', 'analysis', 'epochs', 'general', 'stimulus']
        for g in groups:
            self.f.require_group(g)
    
    @staticmethod
    def resolve_hdfname(data= None):
        hdf_name= None
        file_name= data.getFilename()
        system= data.getSystem()
        if system== 'NWB':
                hdf_name= file_name.split('+')[0] 
        if os.path.exists(file_name):
            words= file_name.split(os.sep)
            f_path= os.sep.join(words[0:-1])
            f_name= words[-1]
            if system== 'Axona':
                if isinstance(data, NSpike) or isinstance(data, NLfp):
                    hdf_name= os.sep.join([f_path, f_name.split('.')[0]+'.hdf5'])
                elif isinstance(data, NSpatial):
                    hdf_name= os.sep.join([f_path, '_'.join(f_name.split('.')[0].split('_')[:-1])+'.hdf5'])
            elif system== 'Neuralynx':
                hdf_name= os.sep.join([f_path, f_path.split(os.sep)[-1]+'.hdf5'])
            
        return hdf_name
    
    def resolve_datapath (self, data= None):
        # No resolution for NWB file, this function will not be called if the system== 'NWB'
        path= None
        tag= self.get_file_tag(data)
        
        if isinstance(data, NSpatial):
            path= '/processing/Behavioural/Position'
        elif tag and isinstance(data, NSpike):
            path= '/processing/Shank/'+ tag
        elif tag and isinstance(data, NLfp):
            path= '/processing/Neural Continuous/LFP/'+ tag

        return path

    @staticmethod
    def get_file_tag(data= None):
        # data is one of NSpike or Nlfp instance
        tag = None
        if isinstance(data, NSpike) or isinstance(data, NLfp):
            f_name= data.getFilename()
            system= data.getSystem()
            if system== 'NWB':
                tag= f_name.split('+')[-1].split('/')[-1]
            else:
                name, ext= os.path.basename(f_name).split('.')
                if system== 'Axona':
                    tag= ext
                elif system== 'Neuralynx':
                    tag= name
        return tag
    
    def resolve_analysis_path(self, spike= None, lfp= None):    
        # Each input is an object    
        path= ''    
        if spike and isinstance(spike, NSpike):
            tag=  self.get_file_tag(spike)
            if spike.getSystem()== 'Axona' or not tag.startswith('TT'):
                tag= 'TT'+ tag
            path+= tag + '_SS_'+ str(spike.getUnitNo())
        
        if lfp and isinstance(lfp, NLfp):
            path+= '_'+ self.get_file_tag(lfp)
        
        return path
    
    def save_dataset(self, path=None, name= None,  data= None, create_group= True):
        # Path is an abosulte path of the group where the dataset will be stored
        # Abosulte path for the dataset will be created using the group path and the name of the dataset
        # If create_group= False (default), it will check if the path exists. If not error will be generated
        # If creat_group= True, it will create the group]\
        if not path:
            logging.error('Invalid group path specified!')
        if not name:
            logging.error('Please provide a name for the dataset!')
        if (path in self.f) or create_group:
           g= self.f.require_group(path)
           if name in g:
               del g[name]
           if isinstance(data, list):
               data= np.array(data)            
           g.create_dataset(name, data= data)
        else:
            logging.error('hdf5 file path can be created or restored!')
            
    def get_dataset(self, group= None, path= '', name= ''):
        if isinstance(group, h5py.Group):
            g= group
        else:
            g= self.f
        if path in g:
            if isinstance(g[path], h5py.Dataset):
                return np.array(g[path])
            elif isinstance(g[path], h5py.Group):
                g= g[path]
                if name in g:
                    return np.array(g[name])
                else:
                    logging.error('Specify a valid name for the required dataset')
        elif name in g:
            return np.array(g[name])
        else:
            logging.error(path + ' not found!'+ 'Specify a valid path or name or check if a proper group is specified!')
    
#    def delete_group_data(self, path= None):
#        # Deletes everything within a group, not the group itself
#        if path in self.f:
#            g= self.f[path]
#        if g.keys():
#            for node in g.keys():
#                del self.f[path+ '/'+ node]
            
    def save_dict_recursive(self, path= None, name= None, data= None, create_group= True):
        if not isinstance(data, dict):
            logging.error('Nhdf class method save_dict_recursive() takes only dictionary data input!')
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    self.save_dict_recursive(path= path+ name+ '/', name= key, data= data[key], create_group= True)
                else:
                    self.save_dataset(path= path+ name, name= key, data= value, create_group= True)
                    
    def save_attributes(self, path= None, attr= None):
        # path has to be the absolute path of a group
        if path in self.f:
            g= self.f[path]           
            if isinstance(attr, dict):                
                for key, val in attr.items():
                    g.attrs[key]= val         
            else:
                logging.error('Please specify the attributes in a dictionary!')    
        else:
            logging.error('Please provide a valid hdf5 path!')
            
    def save_spatial(self, spatial= None):
        if isinstance(spatial, NSpatial):
            # derive the path from the filename to ensure uniqueness
            self.setFilename(self.resolve_hdfname(data= spatial))
            # Get the lfp data path/group
            path= self.resolve_datapath(data= spatial)
            # delete old data
            if path in self.f:
                del self.f[path]
            
            # Create group afresh
            g= self.f.require_group(path)            
            
            self.save_attributes(path= path, attr= spatial.getRecordInfo())
            
            g_loc= g.require_group(path+ '/'+ 'location')
            g_dir= g.require_group(path+ '/'+ 'direction')
            g_speed= g.require_group(path+ '/'+ 'speed')
            g_angVel= g.require_group(path+ '/'+ 'angular velocity')
            
            loc= np.empty((spatial.getTotalSamples(), 2))
            loc[:, 0]= spatial.getX()
            loc[:, 1]= spatial.getY()
            
            g_loc.create_dataset(name= 'data', data= loc)
            g_loc.create_dataset(name= 'num_samples', data= spatial.getTotalSamples())
            g_loc.create_dataset(name= 'timestamps', data= spatial.getTime())
#            g_loc.create_dataset(name= 'unit', data= spatial.getUnit(var= 'speed')) # Unit information needs to be included
            # need to implement the spatial.getUnit() method
            
            g_dir.create_dataset(name= 'data', data= spatial.getDirection())
            g_dir.create_dataset(name= 'num_samples', data= spatial.getTotalSamples())
            g_dir.create_dataset(name= 'timestamps', data= spatial.getTime())
#            g_dir.create_dataset(name= 'timestamps', data= h5py.SoftLink(g_loc.name+ '/timestamps'))
            
            g_speed.create_dataset(name= 'data', data= spatial.getSpeed())
            g_speed.create_dataset(name= 'num_samples', data= spatial.getTotalSamples())
            g_speed.create_dataset(name= 'timestamps', data= spatial.getTime())

            g_angVel.create_dataset(name= 'data', data= spatial.getAngVel())
            g_angVel.create_dataset(name= 'num_samples', data= spatial.getTotalSamples())
            g_angVel.create_dataset(name= 'timestamps', data= spatial.getTime())                        
            
        else:
            logging.error('Please specify a valid NSpatial object')

    def save_lfp(self, lfp= None):
        if isinstance(lfp, NLfp):
            # derive the path from the filename to ensure uniqueness
            self.setFilename(self.resolve_hdfname(data= lfp))
            # Get the lfp data path/group
            path= self.resolve_datapath(data= lfp)            
            # delete old data
            if path in self.f:
                del self.f[path]
            
            # Create group afresh
            g= self.f.require_group(path)
            
            self.save_attributes(path= path, attr= lfp.getRecordInfo())            
            
            g.create_dataset(name= 'data', data= lfp.getSamples())
            g.create_dataset(name= 'num_samples', data= lfp.getTotalSamples())
            g.create_dataset(name= 'timestamps', data= lfp.getTimestamp())
                        
        else:
            logging.error('Please specify a valid NLfp object')
    
    def save_spike(self, spike= None):
        if isinstance(spike, NSpike):
            # derive the path from the filename to ensure uniqueness
            self.setFilename(self.resolve_hdfname(data= spike))
            # Get the spike data path/group
            path= self.resolve_datapath(data= spike)
            
            # delete old data
            if path in self.f:
                del self.f[path]
            
            # Create group afresh
            g= self.f.require_group(path)
            
            self.save_attributes(path= path, attr= spike.getRecordInfo())
            
            g_clust= g.require_group(path+ '/'+ 'Clustering')
            g_wave= g.require_group(path+ '/'+ 'EventWaveForm/WaveForm')
            
            # From chX dictionary, create a higher order np array
            
            waves= spike.getWaveform() # NC waves are stroed in waves['ch1'], waves['ch2'] etc. ways
            stacked_channels= np.empty((spike.getTotalSpikes(), spike.getSamplesPerSpike(), spike.getTotalChannels()))
            i= 0
            for key, val in waves.items():
                stacked_channels[:, :, i]= val
                i+=1
            g_wave.create_dataset(name= 'data', data= stacked_channels)
            g_wave.create_dataset(name= 'electrode_idx', data= spike.getChannelIDs())
            g_wave.create_dataset(name= 'num_events', data= spike.getTotalSpikes())
            g_wave.create_dataset(name= 'num_samples', data= spike.getSamplesPerSpike())
            g_wave.create_dataset(name= 'timestamps', data= spike.getTimestamp())

            # save Clutser number            
            g_clust.create_dataset(name= 'cluster_nums', data= spike.getUnitList())
            g_clust.create_dataset(name= 'num', data= spike.getUnitTags())
            g_clust.create_dataset(name= 'times', data= spike.getTimestamp())
        else:
            logging.error('Please specify a valid NSpike object')
            
    def save_cluster(self, clust= None):
        # Nclust is a NSpike derivative (inherited from NSpike) to add clustering facilities to the NSpike data
        # But we will consider putting it within NSpike itself
        # This will store data to Shank's Clustering and Feature Extraction group
        if isinstance(clust, NClust):
            pass
        else:
            logging.error('Please specify a valid NClust object')
            
    def load_spike(self, path= None, spike= None):
        if not isinstance(spike, NSpike):
            spike= NSpike()
        _recordInfo= {}
                
        if path in self.f:
            g= self.f[path]
        elif '/processing/Shank/'+ path in self.f:
            path= '/processing/Shank/'+ path
            g= self.f[path]
        else:
            logging.error('Specified path does not exist!')
        
        for key, value in g.attrs.items():
            _recordInfo[key]= value
        spike.setRecordInfo(_recordInfo)
        
        path_clust= 'Clustering'
        path_wave= 'EventWaveForm/WaveForm'
        
        if path_clust in g:
            g_clust= g[path_clust]
            spike._setTimestamp(self.get_dataset(group= g_clust, name= 'times'))
            spike.setUnitTags(self.get_dataset(group= g_clust, name= 'num'))
            spike._setUnitList()
        else:
            logging.error('There is no /Clustering in the :' +path)
        
        if path_wave in g:
            g_wave= g[path_wave]
            spike._setTotalSpikes(self.get_dataset(group= g_wave, name= 'num_events'))
            chanIDs= self.get_dataset(group= g_wave, name= 'electrode_idx')
            spike._setChannelIDs(chanIDs)
            
            spike_wave= oDict()
            data= self.get_dataset(group= g_wave, name= 'data')
            if len(data.shape)== 2:
                num_events, num_samples= data.shape
                tot_chans= 1
            elif len(data.shape)== 3:
                num_events, num_samples, tot_chans= data.shape
            else:
                logging.error(path_wave+ '/data contains for more than 3 dimensions!')
                
            if not num_events== self.get_dataset(group= g_wave, name= 'num_events'):
                logging.error('Mismatch between num_events and 1st dimension of '+ path_wave+ '/data')
            if not num_samples== self.get_dataset(group= g_wave, name= 'num_samples'):
                logging.error('Mismatch between num_samples and 2nd dimension of '+ path_wave+ '/data')
            for i in np.arange(tot_chans):
                spike_wave['ch'+ str(i+1)]= data[:, :, i]
            spike._setWaveform(spike_wave)
        else:
            logging.error('There is no /EventWaveForm/WaveForm in the :' +path)
        
        return spike

    def load_lfp(self, path= None, lfp= None):
        if not isinstance(lfp, NLfp):
            lfp= NLfp()
        _recordInfo= {}

        if path in self.f:
            g= self.f[path]
        elif '/processing/Neural Continuous/LFP/'+ path in self.f:
            path= '/processing/Neural Continuous/LFP/'+ path
            g= self.f[path]
        else:
            logging.error('Specified path does not exist!')

        for key, value in g.attrs.items():
            _recordInfo[key]= value
        lfp.setRecordInfo(_recordInfo)
        lfp._setSamples(self.get_dataset(group= g, name= 'data'))
        lfp._setTimestamp(self.get_dataset(group= g, name= 'timestamps'))
        lfp._setTotalSamples(self.get_dataset(group= g, name= 'num_samples'))
        
        return lfp
    
    def load_spatial(self, path= None, spatial= None):
        if not isinstance(spatial, NSpatial):
            spatial= NSpatial()
        _recordInfo= {}
        
        if path in self.f:
            g= self.f[path]
        elif '/processing/Behavioural/Position' in self.f:
            path= '/processing/Behavioural/Position'
            g= self.f[path]
            logging.info('Path for spatial data set to: ' + path)
        else:
            logging.error('Path for spatial data does not exist!')

        for key, value in g.attrs.items():
            _recordInfo[key]= value
        
        if path+ '/'+ 'location' in g:
            g_loc= g[path+ '/'+ 'location']
            data= self.get_dataset(group= g_loc, name= 'data')
            spatial._setPosX(data[:, 0])
            spatial._setPosY(data[:, 1])
            spatial._setTime(self.get_dataset(group= g_loc, name= 'timestamps'))
        else:
            logging.error('Spatial location information not found!')
        
        if path+ '/'+ 'direction' in g:
            g_dir= g[path+ '/'+ 'direction']
            data= self.get_dataset(group= g_dir, name= 'data')
            spatial._setDirection(data)
        else:
            logging.error('Spatial direction information not found!')
            
        if path+ '/'+ 'speed' in g:
            g_speed= g[path+ '/'+ 'speed']
            data= self.get_dataset(group= g_speed, name= 'data')
            spatial._setSpeed(data)
        else:
            logging.error('Spatial speed information not found!')
            
        if path+ '/'+ 'angular velocity' in g:
            g_angVel= g[path+ '/'+ 'angular velocity']
            data= self.get_dataset(group= g_angVel, name= 'data')
            spatial.setAngVel(data)
        else:
            spatial.setAngVel(np.array([]))
            logging.warning('Spatial angular velocity information not found, will be calculated from direction!')
        
        return spatial
            
class NAbstract(object):
    def __init__(self, **kwargs):
        self._filename= kwargs.get('filename', '')
        self._system= kwargs.get('system', 'Axona')
        self._name= kwargs.get('name', 'c0')
        self._description= ''
        self._results= collections.OrderedDict()
        self._recordInfo= { 'File version': '',
          'Date': '',
          'Time': '',
          'Experimenter': '',
          'Comments': '',
          'Duration': 0,
          'Format': 'Axona',
          'Source': self._filename}
        
    def setFilename(self, filename= None):
        if filename is not None: 
            self._filename= filename
    def getFilename(self):
        return self._filename

    def setSystem(self, system= None):
        if system is not None: 
            self._system= system
            
    def getSystem(self):
        return self._system
        
    def setName(self, name= ''):
        self._name= name
    def getName(self):
        return self._name

    def setDescription(self, description= ''):
        self._description= description

    def save_to_hdf5(self, parent_dir):
        pass #implement for each type
    def load(self):
        pass
    @classmethod
    def _newInstance(cls, obj= None, **kwargs):
        if obj is None:
            new_obj= cls(**kwargs)
        elif isinstance(obj, cls):
            new_obj= obj
        elif inspect.isclass(obj):
                cls= obj
        new_obj= cls(**kwargs)
        return new_obj
        
    def getResults(self):
        return self._results    
    def updateResult(self, newResult= {}):
        self._results.update(newResult)
    def resetResults(self):
        self._results= oDict()
        
    def _setFileVersion(self, version= ''):
        self._recordInfo['File version']= version
    def _setDate(self, date_str= ''):
        self._recordInfo['Date']= date_str
    def _setTime(self, time= ''):
        self._recordInfo['Time']= time
    def _setExperiemnter(self, experimenter= ''):
        self._recordInfo['Experimenter']= experimenter
    def _setComments(self, comments= ''):
        self._recordInfo['Comments']= comments
    def _setDuration(self, duration= ''):
        self._recordInfo['Duration']= duration
    def _setSourceFormat(self, system= 'Axona'):
        self._recordInfo['Format']= system
    def _setDataSource(self, filename= None):
        self._recordInfo['Source']= filename

    def getFileVersion(self):
        return self._recordInfo['File version']
    def getDate(self):
        return self._recordInfo['Date']
    def getTime(self, time):
        return self._recordInfo['Time']
    def getExperiemnter(self):
        return self._recordInfo['Experimenter']
    def getComments(self):
        return self._recordInfo['Comments']
    def getDuration(self):
        return self._recordInfo['Duration']
    def getSourceFormat(self, system= 'Axona'):
        return self._recordInfo['Format']
    def getDataSource(self, filename= None):
        return self._recordInfo['Source']
    
    def setRecordInfo(self, new_info= {}):
        self._recordInfo.update(new_info)
        
    def getRecordInfo(self, record_name= None):
        if record_name is None:
            return self._recordInfo
        else:
            return self._recordInfo.get(record_name, None)

class NBase(NAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)                
        self._spikes= []
        self._spikesByName= collections.OrderedDict()
        self._lfp= []
        self._lfpByName= collections.OrderedDict()        

        self._recordInfo= { 'File version': '',
          'Date': '',
          'Time': '',
          'Experimenter': '',
          'Comments': '',
          'Duration': 0,
          'No of channels': 1,
          'Bytes per timestamp': 1,
          'Sampling rate': 1,
          'Bytes per sample': 1,
          'ADC Fullscale mv': 1,
          'Format': 'Axona',
          'Source': self._filename}
          
    def addNode(self, node, node_type= None, **kwargs):
        name= node.getName()
        _replace= kwargs.get('replace', False)      

        if node_type is None:
            logging.error('Node type is not defined')
        elif node_type== 'spike':
            nodeNames= self.getSpikeNames()
            nodes= self._spikes
            nodesByName= self._spikesByName
        elif node_type=='lfp':
            nodeNames= self.getLfpNames()
            nodes= self._lfp
            nodesByName= self._lfpByName
        
        if _replace:
            i= self.delNode(node)
        elif name in nodeNames:
            logging.warning(node_type + ' with name {0} already exists, '.format(name) + \
                        'cannot add another one.\r\n' + \
                        'Try renaming or set replace True')
        else:
            i= len(nodes)
            
        nodes.insert(i, node)
        nodesByName[ name ]= node
                      
    def delNode(self, node):
        i= None
        if node in self._spikes:
            i= self._spikes.index(node)
            self._spikes.remove(node)
            del self._spikesByName[ node.getName()]
        elif node in self._lfp:
            i= self._lfp.index(node)
            self._lfp.remove(node)
            del self._lfpByName[ node.getName()]
        return i
        
    def getNode(self, node_names, node_type= 'spike'):
        nodes= []
        not_nodes= []
        if node_type== 'spike':
            names= self.getSpikeNames()
            nodes= self._spikesByName
        elif node_type== 'lfp':
            names= self.getLfpNames()
            nodes= self._lfpByName
        for name in node_names:
            nodes.append(nodes[name]) if name in names\
                else not_nodes.append(name)
        if not_nodes:
            logging.warning(','.join(not_nodes)+ ' does not exist')
        return nodes
        
    def getSpike(self, names= None):
        if names is None:
            spikes= self._spikes
        else:
            spikes= self.getNode(names, 'spike')
        return spikes

    def getLfp(self, names= None):
        if names is None:
            lfp= self._lfp
        else:
            lfp= self.getNode(names, 'lfp')
        return lfp

    def delSpike(self, spike):
        if isinstance(spike, str):
            name= spike
            spike= self.getSpikesByName(name)
        i= self.delNode(spike)
        return i

    def delLfp(self, lfp):
        i= 0
        if isinstance(lfp, str):
            name= lfp
            lfp= self.getLfpByName(name)
        i= self.delNode(lfp)
        return i        
        
    def getSpikeNames(self):
        return self._spikesByName.keys()

    def getLfpNames(self):
        return self._lfpByName.keys()
    
    def changeNames(self, old_names, new_names, node_type= 'spike'):       
        if len(new_names) != len(old_names):
            logging.error('Input names are not equal in numbers!')
        elif len(set(new_names))< len(old_names):
            logging.error('Duplicate names are not allowed!')
        else:
            if node_type== 'spike':
                for i, name in enumerate(new_names):
                    node= self.getSpike(old_names[i])
                    node.setName(name);
                    self._spikeByName[name] = self._spikeByName.pop(old_names[i])
            
            elif node_type== 'lfp':
                for i, name in enumerate(new_names):
                    node= self.getLfp(old_names[i])
                    node.setName(name);
                    self._lfpByName[name] = self._spikeByName.pop(old_names[i])

    def setSpikeNames(self, names):
        self.changeNames(self, self.getSpikeNames(), names, 'lfp')

    def setLfpNames(self, names):
        self.changeNames(self, self.getLfpNames(), names, 'lfp')
        
    def setNodeFileNames(self, node_names, filenames, node_type= 'spike'):
        if len(node_names)!= len(filenames):
            logging.error('No. of names does not match with no. of filenames')
        elif len(set(node_names))!= len(node_names):
            logging.error('Duplicate names are not allowed!')
        else:
            nodes= self.getNode(node_names, node_type)
            for node in nodes:
                node.setFilename(filenames(nodes.index(node)))
                
    def setSpikeFilenames(self, spike_names, filenames):
        self.setNodeFilenames(spike_names, filenames, 'spike')
        
    def setLfpFilenames(self, lfp_names, filenames):
        self.setNodeFilenames(lfp_names, filenames, 'lfp')
        
    def countSpike(self):
        return len(self._spikes)

    def countLfp(self):
        return len(self._lfp)
            
    def _addNode(self, cls, node, node_type, **kwargs):
        new_node= self._newInstance(cls, node, **kwargs)
        self.addNode(new_node, node_type, replace= kwargs.get('replace', False))
        return new_node
        
    def _getInstance(self, cls, node, node_type):
        if isinstance(node, cls):
            new_node= node
        else: 
            if node_type== 'lfp':
                _getNodeNames= self.getLfpNames
                _getNode= self.getLfp
            if node_type== 'spike':
                _getNodeNames= self.getSpikeNames
                _getNode= self.getSpike
            if node in _getNodeNames():
                new_node= _getNode(node)
        return new_node
        
class NEvent(NBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._eventNames=[]
        self._eventTags= []
        self._currTag= []
        self._currName= []
        self._timestamp= np.array([], dtype= np.float32)
        self._eventTrain= np.array([], dtype= int)
        self._type= 'event'
    
    def getEventName(self, event_tag= None):
        if event_tag is None:
            event_name= self._eventNames
        elif event_tag in self._eventTags:
            event_name= self._eventNames[self._eventTags.index(event_tag)]
        else: 
            event_name= None
        return event_name
    
    def getTag(self, event_name= None):
        if event_name is None:
            event_tag= self._currTag
        elif event_name== 'all':
            event_tag= self._eventTags
        elif event_name in self._eventNames:
            event_tag= self._eventTags[self._eventNames.index(event_name)]
        else:
            event_tag= None    
        return event_tag

    def setCurrTag(self, event):
        if event is None:
            pass
        elif event in self._eventTags:
            self._currTag= event
            self._currName= self.getEventName(event)
        elif event in self._eventNames:
            self._currTag= self.getTag(event)
            self._currName= event            
    def setCurrName(self, name):
        self.setCurrTag(self, name)            
            
    def getEventStamp(self, event= None):
        if event is None:
            event= self._currTag
        if event in self._eventNames:
            tag= self.getTag(event) 
        elif event in self._eventTags:
            tag= event
        timestamp= self._timestamp[self._eventTrain== tag]
        return timestamp
    
    def getTimeStamp(self):
        return self._timestamp   
    def _setTimestamp(self, timestamp):
        self._timestamp= timestamp
    def getEventTrain(self):
        return self._eventTrain
    def _setEventTrain(self, eventTrain):
        self._eventTrain= eventTrain
        
    def load(self, filename= None, system= None):
        if system is None:
            system= self._system
        if filename is None:
            filename= self._filename
            
    def _createTag(self, nameTrain):
        self._eventTags= list(range(0, len(self._eventNames), 1))
        if type(nameTrain).__module__ == np.__name__:
            self._eventTrain= np.zeros(nameTrain.shape)
            for i, name in enumerate(self._eventNames):
                self._eventTrain[nameTrain== name]= self._eventTags[i]
                
    def addSpike(self, spike= None, **kwargs):
        new_spike= self._addNode(NSpike, spike, 'spike', **kwargs)
        return new_spike
                
    def loadSpike(self, names= 'all'):
        if names== 'all':
            for spike in self._spikes:
                spike.load()
        else:
            for name in names:
                spike= self.getSpikesByName(name)
                spike.load()
            
    def addLfp(self, lfp= None, **kwargs):
        new_lfp= self._addNode(NLfp, lfp, 'lfp', **kwargs)
        return new_lfp
        
    def loadLfp(self, names= None):
        if names is None:
            self.load()
        elif names== 'all':
            for lfp in self._lfp:
                lfp.load()
        else:
            for name in names:
                lfp= self.getLfpByName(name)
                lfp.load()                   
                
    def psth(self, event= None, spike= None, **kwargs):        
        graphData= oDict()
        if not event:
            event= self._currTag         
        elif event in self._eventNames:
           event= self.getTag(event)
        if not spike:
            spike= kwargs.get('spike', 'xxxx')
        spike= self.getSpike(spike)
        if event:
            if spike:
                graphData= spike.psth(self.getEventStamp(event), **kwargs)
            else:
                logging.error('No valid spike specified')              
        else:
            logging.error(str(event) + ' is not a valid event')
            
        return graphData
       
        # Do things on event tag
     #def other functions; check Pradeep's code
     
    def phaseDist(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')
            _lfp.phaseDist(self.getEventStamp(self.getTag()), **kwargs)
    
    def PLV(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')
            _lfp.PLV(self.getEventStamp(self.getTag()), **kwargs)
    
    def SFC(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')        
            _lfp.SFC(self.getEventStamp(self.getTag()), **kwargs)        
     # NSpike class definition        
class NSpike(NBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 
        self._unitNo= kwargs.get('unitNo', 0)
        self._unitStamp= []
        self._timestamp= []
        self._unitList= []
        self._unitTags= []
        self._waveform= []
        self._type= 'spike'
        self.setRecordInfo({'Timebase': 1,
                            'Samples per spike': 1,
                            'No of spikes': 0,
                            'Channel IDs': None})
    
    def getUnitTags(self):
        return self._unitTags
    def setUnitTags(self, new_tags):
        if len(new_tags)== len(self._timestamp):
            self._unitTags= new_tags
            self._setUnitList()
        else:
            logging.error('No of tags spikes does not match with no of spikes')
    
    def getUnitList(self):
        return self._unitList
    def _setUnitList(self):
        self._unitList= list(map(int, set(self._unitTags)))
        if 0 in self._unitList:
            self._unitList.remove(0)
        
    def setUnitNo(self, unit_no= None, spike_name= None):
        if isinstance(unit_no, int):
            if unit_no in self.getUnitList():
                self._unitNo= unit_no
                self._setUnitStamp()
        else: 
            if spike_name is None:
                spike_name= self.getSpikeNames()
            if len(unit_no)== len(spike_name):
                spikes= self.getSpike(spike_name)
                for i, num in enumerate(unit_no):
                    if num in spikes[i].getUnitList():
                        spikes[i].setUnitNo(num)
            else:
                logging.error('Unit no. to set are not as many as child spikes!')
                
    def getUnitNo(self, spike_name= None):
        if spike_name is None:
            unit_no= self._unitNo
        else:
            unit_no= []
            spikes= self.getSpike(spike_name)
            for spike in spikes:
                unit_no.append(spike._unitNo)
        return unit_no
        
    def getTimestamp(self, unitNo= None):
        if unitNo is None:
            return self._timestamp
        else:
            if unitNo in self._unitList:
                return self._timestamp[self._unitTags== unitNo]
            else:
                logging.warning('Unit ' + str(unitNo) + ' is not present in the spike data')
    
    def _setTimestamp(self, timestamp= None):
        if timestamp is not None:
            self._timestamp= timestamp
            
    def getUnitStamp(self):
        return self.getTimestamp(self._unitNo)
        
    def _setUnitStamp(self):
        self._unitStamp= self.getUnitStamp()
    
    def getUnitSpikesCount(self, unit_no= None):
        if unit_no is None:
            unit_no= self._unitNo
        if unit_no in self._unitList:  
            return sum(self._unitTags== unit_no)

    def getWaveform(self):
        return self._waveform
        
    def _setWaveform(self, spike_waves= []):
        if spike_waves:
            self._waveform= spike_waves      

    def getUnitWaves(self, unit_no= None):
        if unit_no is None:
            unit_no= self._unitNo
        _waves= oDict()
        for chan, wave in self._waveform.items():
            _waves[chan]= wave[self._unitTags== unit_no, :]
        return _waves
    
    # For multi-unit analysis, {'SpikeName': cell_no} pairs should be used as function input
                
    def load(self, filename= None, system= None):
        if system is None:
            system= self._system
        else:
            self._system= system
        if filename is None:
            filename= self._filename
        else:
            filename= self._filename
        loader= getattr(self, 'loadSpike_'+ system)
        loader(filename)
            
    def addSpike(self, spike= None, **kwargs):
        new_spike= self._addNode(self.__class__, spike, 'spike', **kwargs)
        return new_spike
         
    def loadSpike(self, names= None):
        if names is None:
            self.load()
        elif names== 'all':
            for spike in self._spikes:
                spike.load()
        else:
            for name in names:
                spike= self.getSpikesByName(name)
                spike.load()
            
    def addLfp(self, lfp= None, **kwargs):
        new_lfp= self._addNode(NLfp, lfp, 'lfp', **kwargs)
        return new_lfp

    def loadLfp(self, names= 'all'):
        if names== 'all':
            for lfp in self._lfp:
                lfp.load()
        else:
            for name in names:
                lfp= self.getLfpByName(name)
                lfp.load()
    
    def waveProperty(self):
        
        _result= oDict()
        graphData= {}
            
        def argpeak(data):
            data= np.array(data)
            peak_loc= [j for j in range(7, len(data)) \
                        if data[j] <= 0 and data[j- 1] > 0]
            return peak_loc[0] if peak_loc else 0
            
        def argtrough2(data, peak_loc):
            data= data.tolist()
            trough_loc= [j for j in range(7, len(data)) \
                        if data[j]>= 0 and data[j-1]<0 and peak_loc>0]
            return trough_loc[0] if trough_loc else 0
            
        def argtrough1(data, peak_loc):
            data= data.tolist()
            trough_loc= [peak_loc-j for j in range(peak_loc-2) \
                        if data[peak_loc- j] >= 0 and data[peak_loc-j-1] <= 0]
            return trough_loc[0] if trough_loc else 0
        def waveWidth(wave, peak, thresh= 0.25):
            p_loc, p_val= peak
            Len= wave.size
            if p_loc:
                w_start= find(wave[:p_loc] <= thresh*p_val, 1, 'last')  
                w_start= w_start[0] if w_start.size else 0
                w_end= find(wave[p_loc:] <= thresh*p_val, 1, 'first')
                w_end= p_loc+ w_end[0] if w_end.size else Len
            else:
                w_start= 1
                w_end= Len
            
            return w_end- w_start            
        
        num_spikes= self.getUnitSpikesCount()
        _result['Mean Spiking Freq']= num_spikes/ self.getDuration()
        _waves= self.getUnitWaves()
        samplesPerSpike= self.getSamplesPerSpike()
        tot_chans= self.getTotalChannels()
        meanWave= np.empty([samplesPerSpike, tot_chans])
        stdWave= np.empty([samplesPerSpike, tot_chans])
        
        width= np.empty([num_spikes, tot_chans])
        amp= np.empty([num_spikes, tot_chans])
        height= np.empty([num_spikes, tot_chans])
        i= 0
        for chan, wave in _waves.items():
            meanWave[:, i]= np.mean(wave, 0)
            stdWave[:, i]= np.std(wave, 0)
            slope= np.diff(wave)
            max_val= wave.max(1)
            
            if max_val.max()> 0:
                peak_loc= [argpeak(slope[I, :]) for I in range(num_spikes)]
                peak_val= [wave[I, peak_loc[I]] for I in range(num_spikes)]
                trough1_loc= [argtrough1(slope[I, :], peak_loc[I]) for I in range(num_spikes)]
                trough1_val= [wave[I, trough1_loc[I]] for I in range(num_spikes)]
                peak_loc= np.array(peak_loc)
                peak_val= np.array(peak_val)
                trough1_loc= np.array(trough1_loc)
                trough1_val= np.array(trough1_val)
                width[:, i]= np.array([waveWidth(wave[I, :], (peak_loc[I], peak_val[I]), 0.25) \
                             for I in range(num_spikes)])

            amp[:, i]= peak_val- trough1_val
            height[:, i]= peak_val- wave.min(1)

            i+=1
            
        max_chan= amp.mean(0).argmax()
        width= width[:, max_chan]* 10**6/self.getSamplingRate()
        amp= amp[:, max_chan]
        height= height[:, max_chan]
        
        graphData = {'Mean wave': meanWave, 'Std wave': stdWave,
                     'Amplitude': amp, 'Width': width, 'Height': height}
        
        _result.update({'Mean amplitude': amp.mean(), 'Std amplitude': amp.std(),
                 'Mean height': height.mean(), 'Std height': height.std(),
                'Mean width': width.mean(), 'Std width': width.std()})
        
        self.updateResult(_result)
        return graphData
        
    def isi(self, bins= 'auto', bound= None, density= False):
        graphData= oDict()
        unitStamp= self.getUnitStamp()
        isi= 1000*np.diff(unitStamp)
        
        graphData[ 'isiHist' ], edges= np.histogram(isi, bins= bins, range= bound, density= density) 
        graphData[ 'isiBins' ]= edges[:-1]
        graphData[ 'isi' ]= isi
        graphData['maxCount']= graphData[ 'isiHist' ].max()
        graphData['isiBefore']= isi[:-1]
        graphData['isiAfter']= isi[1:]
        
        return graphData
    
    def isiCorr(self, spike= None, **kwargs):
        graphData= oDict()
        if spike is None:
            _unitStamp= np.copy(self.getUnitStamp())
        elif isinstance(spike, int):
            if spike in self.getUnitList():
                _unitStamp= self.getTimeStamp(spike)
        else:
            if isinstance(spike, str):
                spike=  self.getSpike(spike)
            if isinstance(spike, self.__class__):
                _unitStamp= spike.getUnitStamp()
            else:
                logging.error('No valid spike specified')
                
        _corr = self.psth(_unitStamp, **kwargs)
        graphData['isiCorrBins']= _corr['bins']
        center= find(_corr['bins']==0, 1, 'first')[0]
        graphData['isiCorr']= _corr['psth']
        graphData['isiCorr'][center]= graphData['isiCorr'][center] \
                                    - np.min([self.getUnitStamp().size, _unitStamp.size]) 

        return graphData
        
    def psth(self, eventStamp, **kwargs):
        graphData= oDict()
        bins= kwargs.get('bins', 1)
        if isinstance(bins, int):
            bound= np.array(kwargs.get('bound', [-500, 500]))
            bins= np.hstack((np.arange(bound[0] , 0, bins),np.arange(0, bound[1]+ bins, bins)))       
        bins= bins/1000 # converted to sec
        n_bins= len(bins)-1
            
        hist_count= np.zeros([n_bins, ])
        unitStamp= self.getUnitStamp()
        for it in range(eventStamp.size):
            tmp_count, edges= np.histogram(unitStamp- eventStamp[it], bins= bins)
            hist_count= hist_count+ tmp_count
            
        graphData['psth']= hist_count
        graphData['bins']= 1000*edges[:-1]

        return graphData
        
    def burst(self, burstThresh= 5, ibiThresh= 50):
        _results= oDict()
        
        unitStamp= self.getUnitStamp()
        isi= 1000*np.diff(unitStamp)
        
        burst_start=[]
        burst_end= []
        burst_duration=[]
        spikesInBurst= []
        bursting_isi= []
        num_burst= 0
        ibi= []
        duty_cycle= []
        k=0
        while k < isi.size:
            if isi[k] <= burstThresh:
                burst_start.append(k)
                spikesInBurst.append(2)
                bursting_isi.append(isi[k])
                burst_duration.append(isi[k])
                m= k+1
                while m< isi.size and isi[m] <= burstThresh:
                    spikesInBurst[num_burst] += 1
                    bursting_isi.append(isi[m])
                    burst_duration[num_burst] += isi[m]
                    m +=1
                burst_duration[num_burst] += 1 # to compensate for the span of the last spike
                burst_end.append(m)
                k= m+1
                num_burst +=1
            else:
                k+=1
        if num_burst:
           for j in range(0, num_burst-1):
               ibi.append(unitStamp[burst_start[j+1]]- unitStamp[burst_end[j]])
           duty_cycle= np.divide(burst_duration[1:], ibi)/1000 # ibi in sec, burst_duration in ms
        else:
            logging.warning('No burst detected')

        spikesInBurst= np.array(spikesInBurst) if spikesInBurst else np.array([])
        bursting_isi= np.array(bursting_isi) if bursting_isi else np.array([])
        ibi= 1000*np.array(ibi) if ibi else np.array([]) # in sec unit, so converted to ms
        burst_duration= np.array(burst_duration) if burst_duration else np.array([])
        duty_cycle= np.array(duty_cycle) if len(duty_cycle) else np.array([])
        
        _results['Total burst']=  num_burst
        _results['Total bursting spikes']=  spikesInBurst.sum()
        _results['Mean bursting ISI ms']= bursting_isi.mean() if bursting_isi.any() else None
        _results['Std bursting ISI ms']= bursting_isi.std() if bursting_isi.any() else None
        _results['Mean spikes per burst']= spikesInBurst.mean() if spikesInBurst.any() else None
        _results['Std spikes per burst']= spikesInBurst.std() if spikesInBurst.any() else None
        _results['Mean burst duration ms']= burst_duration.mean() if burst_duration.any() else None
        _results['Std burst duration']= burst_duration.std() if burst_duration.any() else None
        _results['Mean duty cycle']= duty_cycle.mean() if duty_cycle.any() else None
        _results['Std duty cycle']= duty_cycle.std() if duty_cycle.any() else None
        _results['Mean IBI']= ibi.mean() if ibi.any() else None
        _results['Std IBI']= ibi.std() if ibi.any() else None
        _results['Propensity to burst']= spikesInBurst.sum()/ unitStamp.size
        self.updateResult(_results)
        
    def thetaIndex(self, **kwargs):
        p_0= kwargs.get('start', [6, 0.1, 0.05])
        lb= kwargs.get('lower', [4, 0, 0])
        ub=  kwargs.get('upper', [14, 5, 0.1])
        
        _results= oDict()
        graphData= self.isiCorr(**kwargs)
        corrBins= graphData['isiCorrBins']
        corrCount= graphData['isiCorr']
        m= corrCount.max()
        center= find(corrBins==0, 1, 'first')[0]
        x= corrBins[center:]/1000
        y= corrCount[center:]
        y_fit= np.empty([corrBins.size,])

## This is for the double-exponent dip model        
#        def fit_func(x, a, f, tau1, b, c1, tau2, c2, tau3):
#            return  a*np.cos(2*np.pi*f*x)*np.exp(-np.abs(x)/tau1)+ b+ \
#                c1*np.exp(-np.abs(x)/tau2)- c2*np.exp(-np.abs(x)/tau3)
#
#        popt, pcov = curve_fit(fit_func, x, y, \
#                                p0= [m, p_0[0], p_0[1], m, m, p_0[2], m, 0.005], \
#                                bounds= ([0, lb[0], lb[1], 0, 0, lb[2], 0, 0], \
#                                [m, ub[0], ub[1], m, m, ub[2], m, 0.01]),
#                                max_nfev= 100000)
#        a, f, tau1, b, c1, tau2, c2, tau3= popt

# This is for the single-exponent dip model
        def fit_func(x, a, f, tau1, b, c, tau2):
            return  a*np.cos(2*np.pi*f*x)*np.exp(-np.abs(x)/tau1)+ b+ \
                c*np.exp(-(x/tau2)**2)
                
        popt, pcov = curve_fit(fit_func, x, y, \
                                p0= [m, p_0[0], p_0[1], m, m, p_0[2]], \
                                bounds= ([0, lb[0], lb[1], 0, -m, lb[2]], \
                                [m, ub[0], ub[1], m, m, ub[2]]),
                                max_nfev= 100000)
        a, f, tau1, b, c, tau2= popt 
        
        y_fit[center:]= fit_func(x, *popt)
        y_fit[:center]= np.flipud(y_fit[center:])
        
        gof= residualStat(y, y_fit[center:], 6)
        
        graphData['corrFit']= y_fit        
        _results['Theta Index']= a/b
        _results['TI fit freq Hz']= f
        _results['TI fit tau1 Hz']= tau1
        _results['TI adj Rsq']= gof['adj Rsq']
        _results['TI Pearse R']= gof['Pearson R']
        _results['TI Pearse P']= gof['Pearson P']
        
        self.updateResult(_results)   
        
        return graphData
        
    def thetaSkipIndex(self, **kwargs):            
        p_0= kwargs.get('start', [6, 0.1, 0.05])
        lb= kwargs.get('lower', [4, 0, 0])
        ub=  kwargs.get('upper', [14, 5, 0.1])
        
        _results= oDict()
        graphData= self.isiCorr(**kwargs)
        corrBins= graphData['isiCorrBins']
        corrCount= graphData['isiCorr']
        m= corrCount.max()
        center= find(corrBins==0, 1, 'first')[0]
        x= corrBins[center:]/1000
        y= corrCount[center:]
        y_fit= np.empty([corrBins.size,])

# This is for the double-exponent dip model        
        def fit_func(x, a1, f1, a2, f2, tau1, b, c1, tau2, c2, tau3):
            return  (a1*np.cos(2*np.pi*f1*x)+ a2*np.cos(2*np.pi*f2*x))*np.exp(-np.abs(x)/tau1)+ b+ \
                c1*np.exp(-np.abs(x)/tau2)- c2*np.exp(-np.abs(x)/tau3)

        popt, pcov = curve_fit(fit_func, x, y, \
                                p0= [m, p_0[0], m, p_0[0]/2, p_0[1], m, m, p_0[2], m, 0.005], \
                                bounds= ([0, lb[0], 0, lb[0]/2, lb[1], 0, 0, lb[2], 0, 0], \
                                [m, ub[0], m, ub[0]/2, ub[1], m, m, ub[2], m, 0.01]),
                                max_nfev= 100000)
        a1, f1, a2, f2, tau1, b, c1, tau2, c2, tau3= popt

## This is for the single-exponent dip model
#        def fit_func(x, a1, f1, a2, f2, tau1, b, c, tau2):
#            return  (a1*np.cos(2*np.pi*f1*x)+ a2*np.cos(2*np.pi*f2*x))*np.exp(-np.abs(x)/tau1)+ b+ \
#                c*np.exp(-(x/tau2)**2)
#                
#        popt, pcov = curve_fit(fit_func, x, y, \
#                                p0= [m, p_0[0], m, p_0[0]/2, p_0[1], m, m, p_0[2]], \
#                                bounds= ([0, lb[0], 0, lb[0]/2, lb[1], 0, -m, lb[2]], \
#                                [m, ub[0], m, ub[0]/2, ub[1], m, m, ub[2]]),
#                                max_nfev= 100000)
#        a1, f1, a2, f2, tau1, b, c, tau2= popt 
        
        temp_fit= fit_func(x, *popt)
        y_fit[center:]= temp_fit
        y_fit[:center]= np.flipud(temp_fit)
        
        peak_val, peak_loc= extrema(temp_fit[find(x>= 50/1000)])[0:2]
        
        if peak_val.size>=2:
            skipIndex= (peak_val[1]- peak_val[0])/np.max(np.array([peak_val[1], peak_val[0]]))
        else:
            skipIndex= None
        gof= residualStat(y, temp_fit, 6)
        
        graphData['corrFit']= y_fit        
        _results['Theta Skip Index']= skipIndex
        _results['TS jump factor']= a2/(a1+ a2) if skipIndex else None
        _results['TS f1 freq Hz']= f1 if skipIndex else None
        _results['TS f2 freq Hz']= f2 if skipIndex else None
        _results['TS freq ratio']= f1/f2 if skipIndex else None
        _results['TS tau1 Hz']= tau1 if skipIndex else None
        _results['TS adj Rsq']= gof['adj Rsq']
        _results['TS Pearse R']= gof['Pearson R']
        _results['TS Pearse P']= gof['Pearson P']
        
        self.updateResult(_results)   
        
        return graphData
    
    def phaseDist(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')
            _lfp.phaseDist(self.getTimestamp(self.getUnitNo()), **kwargs)
    
    def PLV(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')
            _lfp.PLV(self.getUnitStamp(), **kwargs)
    
    def SFC(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')        
            _lfp.SFC(self.getUnitStamp(), **kwargs)
        
    def spikeLfpCausality(self, lfp= None, **kwargs):
        if lfp is None:
            logging.error('LFP data not specified!')
        else: 
            _lfp= self._getInstance(NLfp, lfp, 'lfp')        
            _lfp.spikeLfpCausality(self.getUnitStamp(), **kwargs)
 
#    def clusterQuality(self):
#        pass
#    
#    def writeNwb(self):
#        pass
#    
#    def readNwb(self):
#        pass
    
    def _setTotalSpikes(self, spike_count= 1):
        self._recordInfo['No of spikes']= spike_count
        self.spikeCount= spike_count
    def _setTotalChannels(self, tot_channels= 1):
        self._recordInfo['No of channels']= tot_channels
    def _setChannelIDs(self, channel_IDs):
        self._recordInfo['Channel IDs']= channel_IDs
    def _setTimestampBytes(self, bytes_per_timestamp):
        self._recordInfo['Bytes per timestamp']= bytes_per_timestamp
    def _setTimebase(self, timebase= 1):
        self._recordInfo['Timebase']= timebase
    def _setSamplingRate(self, sampling_rate= 1):
        self._recordInfo['Sampling rate']= sampling_rate
    def _setBytesPerSample(self, bytes_per_sample= 1):
        self._recordInfo['Bytes per sample']= bytes_per_sample
    def _setSamplesPerSpike(self, samples_per_spike= 1):
        self._recordInfo['Samples per spike']= samples_per_spike
    def _setFullscaleMv(self, adc_fullscale_mv= 1):
        self._recordInfo['ADC Fullscale mv']= adc_fullscale_mv
    

    def getTotalSpikes(self):
        return self._recordInfo['No of spikes']
    def getTotalChannels(self):
        return self._recordInfo['No of channels']
    def getChannelIDs(self):
        return self._recordInfo['Channel IDs']
    def getTimestampBytes(self):
        return self._recordInfo['Bytes per timestamp']
    def getTimebase(self):
       return  self._recordInfo['Timebase']
    def getSamplingRate(self):
        return self._recordInfo['Sampling rate']
    def getBytesPerSample(self):
        return self._recordInfo['Bytes per sample']
    def getSamplesPerSpike(self):
        return self._recordInfo['Samples per spike']
    def getFullscaleMv(self):
        return self._recordInfo['ADC Fullscale mv']
    
    def save_to_hdf5(self, file_name= None, system= None):
        hdf= Nhdf()
        if file_name and system:
            if os.path.exists(file_name) :
                self.setFilename(file_name)
                self.setSystem(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')                                
        
        hdf.save_spike(spike= self)
        hdf.close()
    
    def loadSpike_NWB(self, file_name):
        file_name, path= file_name.split('+')
        if os.path.exists(file_name):
            hdf= Nhdf()
            hdf.setFilename(file_name)
            self= hdf.load_spike(path= path, spike= self)
            hdf.close()
            
    def loadSpike_Axona(self, file_name):    
        words= file_name.split(sep= os.sep)
        file_directory= os.sep.join(words[0:-1])
        file_tag= words[-1].split(sep= '.')[0]
        tet_no= words[-1].split(sep= '.')[1]
        set_file= file_directory + os.sep + file_tag + '.set'
        cut_file= file_directory + os.sep + file_tag + '_' + tet_no + '.cut'
        
        self._setDataSource(file_name)
        self._setSourceFormat('Axona')
        
        with open(file_name, 'rb') as f:
            while True:
                line= f.readline()
                try:
                    line= line.decode('UTF-8')
                except:
                    break
                    
                if ""== line:
                    break
                if line.startswith('trial_date'):                
                    self._setDate( ' '.join(line.replace(',', ' ').split()[1:]))
                if line.startswith('trial_time'):
                    self._setTime( line.split()[1])
                if line.startswith('experimenter'):
                    self._setExperiemnter( ' '.join(line.split()[1:]))
                if line.startswith('comments'):                
                    self._setComments( ' '.join(line.split()[1:]))
                if line.startswith('duration'):
                    self._setDuration( float(''.join(line.split()[1:])))
                if line.startswith('sw_version'):
                    self._setFileVersion( line.split()[1])
                if line.startswith('num_chans'):
                    self._setTotalChannels( int(''.join(line.split()[1:])))
                if line.startswith('timebase'):
                    self._setTimebase( int(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if line.startswith('bytes_per_timestamp'):
                    self._setTimestampBytes( int(''.join(line.split()[1:])))
                if line.startswith('samples_per_spike'):
                    self._setSamplesPerSpike( int(''.join(line.split()[1:])))
                if line.startswith('sample_rate'):
                    self._setSamplingRate ( int(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if line.startswith('bytes_per_sample'):
                    self._setBytesPerSample( int(''.join(line.split()[1:])))
                if line.startswith('num_spikes'):
                    self._setTotalSpikes( int(''.join(line.split()[1:])))
    
            num_spikes= self.getTotalSpikes()
            bytes_per_timestamp= self.getTimestampBytes()
            bytes_per_sample= self.getBytesPerSample()
            samples_per_spike= self.getSamplesPerSpike()
            
            f.seek(0, 0)
            header_offset= [];
            while True:
                try:
                    buff= f.read(10).decode('UTF-8')
                except:
                    break
                header_offset= f.tell() if buff == 'data_start' else f.seek(-9, 1)
    
            tot_channels= self.getTotalChannels()
            self._setChannelIDs ([(int(tet_no)-1)*tot_channels+ x for x in range(tot_channels)])
            max_ADC_count= 2**(8*bytes_per_sample-1)-1
            max_byte_value= 2**(8*bytes_per_sample) 
            
            with open(set_file, 'r') as f_set:
                lines= f_set.readlines()
                gain_lines= dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                            for line in lines if 'gain_ch_' in line])
                gains= np.array([gain_lines[ch_id] for ch_id in self.getChannelIDs()])      
                for line in lines:
                    if line.startswith('ADC_fullscale_mv'):
                        self._setFullscaleMv( int(re.findall(r'\d+.\d+|d+', line)[0]))
                        break
                AD_bit_uvolts= 2*self.getFullscaleMv()*10**3/ \
                                 (gains*(2**(8*bytes_per_sample)))
    
            record_size= tot_channels*(bytes_per_timestamp + \
                            bytes_per_sample * samples_per_spike)
            time_be=  256**(np.arange(bytes_per_timestamp, 0, -1)-1)
            sample_le= 256**(np.arange(0, bytes_per_sample, 1))
                    
            if not header_offset:
                print('Error: data_start marker not found!')
            else:
                f.seek(header_offset, 0)
                byte_buffer= np.fromfile(f, dtype= 'uint8')                        
                spike_time= np.zeros([num_spikes, ], dtype= 'uint32')
                for i in list(range(0, bytes_per_timestamp)):
                    spike_time= spike_time + time_be[i]*byte_buffer[i:len(byte_buffer)- record_size:record_size]
                spike_time= spike_time/ self.getTimebase() 
                spike_time= spike_time.reshape((num_spikes, ))
                 
                spike_wave= oDict()
    
                
                for i in np.arange(tot_channels):
                    chan_offset= (i+1)*bytes_per_timestamp+ i*bytes_per_sample*samples_per_spike
                    chan_wave= np.zeros([num_spikes, samples_per_spike], dtype= np.float64)
                    for j in np.arange(0, samples_per_spike, 1):
                        sample_offset= j*bytes_per_sample + chan_offset
                        for k in np.arange(0, bytes_per_sample, 1):
                            byte_offset= k + sample_offset
                            sample_value= sample_le[k]* byte_buffer[byte_offset \
                                          : len(byte_buffer)+ byte_offset-record_size\
                                          :record_size]
                            sample_value= sample_value.astype(np.float64, casting= 'unsafe', copy= False)
                            np.add(chan_wave[:, j], sample_value, out= chan_wave[:, j])
                        np.putmask(chan_wave[:, j], chan_wave[:, j]> max_ADC_count, chan_wave[:, j]- max_byte_value)
                    spike_wave['ch'+ str(i+1)]= chan_wave*AD_bit_uvolts[i]
            with open(cut_file, 'r') as f_cut:
                while True:
                    line= f_cut.readline()
                    if ""==line:
                        break
                    if line.startswith('Exact_cut'):
                        unit_ID= np.fromfile(f_cut, dtype= 'uint8', sep= ' ')                           
            self._setTimestamp( spike_time)
            self._setWaveform( spike_wave)
            self.setUnitTags( unit_ID)
        
    def loadSpike_Neuralynx (self, file_name):
#        file_name= 'C:\\Users\\Raju\\Google Drive\\Sample Data for NC\\Sample NLX Data Set\\' + 'TT1.ntt'
#        file_name= 'D:\\Google Drive\\Sample Data for NC\\Sample NLX Data Set\\' + 'TT1.ntt'
        
        self._setDataSource(file_name)
        self._setSourceFormat('Neuralynx')
        
        # Format description for the NLX file:
        file_ext= file_name[-3:]
        if file_ext== 'ntt':
           tot_channels= 4
        elif file_ext== 'nst':
           tot_channels= 2
        elif file_ext== 'nse':
           tot_channels= 1
        header_offset= 16*1024 # fixed for NLX files
        
        bytes_per_timestamp= 8
        bytes_chan_no= 4 
        bytes_cell_no= 4
        bytes_per_feature= 4
        num_features= 8
        bytes_features= bytes_per_feature*num_features
        bytes_per_sample= 2 
        samples_per_record= 32
        channel_pack_size= bytes_per_sample*tot_channels# ch1|ch2|ch3|ch4 each with 2 bytes
        
        max_byte_value= np.power(2, bytes_per_sample*8)
        max_ADC_count= np.power(2, bytes_per_sample*8- 1)-1
        AD_bit_uvolts= np.ones([tot_channels, ])*10**-6 # Default value

        record_size= None
        with open(file_name, 'rb') as f:
            while True:                
                line= f.readline()
                try:
                    line= line.decode('UTF-8')
                except:
                    break
                    
                if ""== line:
                    break
                if 'SamplingFrequency' in line:
                    self._setSamplingRate ( float(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if 'RecordSize' in line:
                    record_size= int(''.join(re.findall(r'\d+.\d+|\d+', line)))
                if 'Time Opened' in line:          
                    self._setDate(re.search(r'\d+/\d+/\d+', line).group())
                    self._setTime(re.search(r'\d+:\d+:\d+', line).group())
                if 'FileVersion' in line:
                    self._setFileVersion(line.split()[1])
                if 'ADMaxValue' in line:
                    max_ADC_count= float(''.join(re.findall(r'\d+.\d+|\d+', line)))  
                if 'ADBitVolts' in line:
                    AD_bit_uvolts= np.array([float(x)*(10**6) for x in re.findall(r'\d+.\d+|\d+', line)])
                if 'ADChannel' in line:
                     self._setChannelIDs(np.array([int(x) for x in re.findall(r'\d+', line)]))
                if 'NumADChannels' in line:
                     tot_channels= int(''.join(re.findall(r'\d+', line)))
                                
            self._setFullscaleMv((max_byte_value/2)*AD_bit_uvolts) # gain= 1 assumed to keep in similarity to Axona
            self._setBytesPerSample(bytes_per_sample)
            self._setSamplesPerSpike(samples_per_record)
            self._setTimestampBytes(bytes_per_timestamp)
            self._setTotalChannels( tot_channels)
            
            if not record_size:
                record_size= bytes_per_timestamp+ \
                             bytes_chan_no+ \
                             bytes_cell_no+ \
                             bytes_features+ \
                             bytes_per_sample*samples_per_record*tot_channels
             
            time_offset= 0
            unitID_offset= bytes_per_timestamp+ \
                           bytes_chan_no
            sample_offset= bytes_per_timestamp+ \
                           bytes_chan_no+ \
                           bytes_cell_no+ \
                           bytes_features
            f.seek(0, 2)
            num_spikes= int((f.tell()- header_offset)/record_size)
            self._setTotalSpikes(num_spikes)
            
            f.seek(header_offset, 0)
            spike_time= np.zeros([num_spikes, ])
            unit_ID= np.zeros([num_spikes, ], dtype= int)
            spike_wave= oDict()
            sample_le= 256**(np.arange(bytes_per_sample))
            for i in np.arange(tot_channels):
                spike_wave['ch'+ str(i+1)]= np.zeros([num_spikes, samples_per_record]) 
            
            for i in np.arange(num_spikes):
                sample_bytes= np.fromfile(f, dtype= 'uint8', count= record_size)
                spike_time[i]= int.from_bytes(sample_bytes[time_offset+ np.arange(bytes_per_timestamp)], byteorder= 'little', signed= False)/10**6
                unit_ID[i]= int.from_bytes(sample_bytes[unitID_offset+ np.arange(bytes_cell_no)], byteorder= 'little', signed= False)
                
                for j in range(tot_channels):
                    sample_value= np.zeros([samples_per_record, bytes_per_sample])
                    ind= sample_offset+ j*bytes_per_sample+ np.arange(samples_per_record)*channel_pack_size
                    for k in np.arange(bytes_per_sample):
                        sample_value[:, k]= sample_bytes[ind+ k]
                    sample_value= sample_value.dot(sample_le)
                    np.putmask(sample_value, sample_value> max_ADC_count, sample_value- max_byte_value)
                    spike_wave['ch'+ str(j+1)][i, :]= sample_value*AD_bit_uvolts[j]
            spike_time-= spike_time.min()
            self._setDuration(spike_time.max())
            self._setTimestamp(spike_time)
            self._setWaveform(spike_wave)
            self.setUnitTags( unit_ID)

class NSpatial(NAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._time= []
        self._pixelSize= 3
        self._posX= []
        self._posY= []
        self._direction= []
        self._speed= []
        self._angVel= []
        self._borderDist= []
        self._xbound= []
        self._ybound= []
        self._distMap= []
        self._type= 'spatial'
        self.spike= []
        self.lfp= []
        
        
    def setPixelSize(self, pixelSize):
        self._pixelSize= pixelSize
    def _setTime(self, time):
        self._time= time
        self._setTimestamp()
        self._setSamplingRate()
    def _setTimestamp(self, timestamp= None):
        if timestamp:
            self._timestamp= timestamp
        elif np.array(self._time).any():
            self._timestamp= np.diff(np.array(self._time)).mean()
    def _setSamplingRate(self, sampling_rate= None):
        if sampling_rate:
            self._Fs= sampling_rate
        elif np.array(self._time).any():
            self._Fs= 1/np.diff(np.array(self._time)).mean()
            
    def _setPosX(self, pos_x):
        self._posX= pos_x
    def _setPosY(self, pos_y):
        self._posY= pos_y
    def _setDirection(self, direction):
        self._direction= direction
    def _setSpeed(self, speed):
        self._speed= speed
    def setAngVel(self, angVel):
        self._angVel= angVel
    def setBorder(self, border):
        self._borderDist= border[0]
        self._xbound= border[1]
        self._ybound= border[2]
        self._distMap= border[3]
    
    def getTotalSamples(self):
        return self._time.size
    def getSamplingRate(self):
        return self._Fs
    def getDuration(self):
        return self._time[-1]
    def getPixelSize(self):
        return self._pixelSize
    def getTime(self):
        return self._time
    def getX(self):
        return self._posX
    def getY(self):
        return self._posY
    def getDirection(self):
        return self._direction
    def getSpeed(self):
        return self._speed
    def getAngVel(self):
        return self._angVel
    def getBorder(self):
        return self._borderDist, self._xbound, self._ybound, self._distMap
        
    def setSpike(self, spike, **kwargs):
        if spike is isinstance(spike, NSpike):
            self.spike= spike
        else:
            cls= NSpike if not spike else spike
            spike= cls(**kwargs)
        self.spike= spike
        
    def setLfp(self, lfp, **kwargs):
        if lfp is isinstance(lfp, NLfp):
            self.lfp= lfp
        else:
            cls= NLfp if not lfp else lfp
            lfp= cls(**kwargs)
        self.lfp= lfp
    
    def setSpikeName(self, name= None):
        if name is not None:
            self.spike.setName(name)
    def setSpikeFilename(self, filename= None):
        if filename is not None:
            self.spike.setFilename()
            
    def setLfpName(self, name= None):
        self.lfp.setName(name)
    def setLfpFilename(self, filename= None):
        self.lfp.setFilename(filename)

    def setEvent(self, event, **kwargs):
        if event is isinstance(event, NEvent):
            self.event= event
        else:
            cls= NEvent if not event else event
            event= cls(**kwargs)
        self.event= event
                
    def setEventName(self, name= None):
        self.event.setName(name)
    def setEventFilename(self, filename= None):
        self.event.setFilename(filename)
        
    def setSystem(self, system= None):
        if system is not None: 
            self._system= system
            
            if self.spike:
                self.spike.setSystem(system)
            if self.lfp:
                self.lfp.setSystem(system)
            
    def loadSpike(self):
        self.spike.load()
    def loadLfp(self):
        self.lfp.load()
        
    def save_to_hdf5(self, file_name= None, system= None):
        hdf= Nhdf()
        if file_name and system:
            if os.path.exists(file_name) :
                self.setFilename(file_name)
                self.setSystem(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')                                
        
        hdf.save_spatial(spatial= self)
        hdf.close()
       
    def load(self, filename= None, system= None):
        if system is None:
            system= self._system
        else:
            self._system= system
        if filename is None:
            filename= self._filename
        else:
            filename= self._filename
        loader= getattr(self, 'loadSpatial_'+ system)
        loader(filename)
        try:
            self.smoothSpeed()
        except:
            logging.warning(self.getSystem() + ' files may not have speed data!')
        if not np.array(self._angVel).any():
            self.setAngVel(self.calcAngVel())
        self.setBorder(self.calcBorder())
        
    def loadSpatial_Axona (self, file_name):
        try:
            f= open(file_name, 'rt')
            self._setDataSource(file_name)
            self._setSourceFormat('Axona')
            while True:
                line= f.readline()
                if ""== line:
                    break
                elif line.startswith('time'):
                    spatial_data= np.loadtxt(f, dtype= 'float', usecols= range(5));
            self._setTime(spatial_data[:, 0])
            self._setPosX(spatial_data[:, 1]- np.min(spatial_data[:, 1]))
            self._setPosY(spatial_data[:, 2]- np.min(spatial_data[:, 2]))
            self._setDirection(spatial_data[:, 3])
            self._setSpeed(spatial_data[:, 4])
            f.seek(0, 0)
            pixel_size= list(map(float, re.findall(r"\d+.\d+|\d+", f.readline())))
            self.setPixelSize(pixel_size)
            self.smoothDirection()
        except:
            logging.error('File does not exist or is open in another process!')

    def loadSpatial_NWB(self, file_name):
        file_name, path= file_name.split('+')
        if os.path.exists(file_name):
            hdf= Nhdf()
            hdf.setFilename(file_name)
            self= hdf.load_spatial(path= path, spatial= self)
            hdf.close()
    def loadSpatial_Neuralynx (self, file_name):
#        file_name= 'C:\\Users\\Raju\\Google Drive\\Sample Data for NC\\Data_sample_events_Rikki\\2014-07-02_62_tts\\' + 'VT1_62_7-02.nvt'
        
        self._setDataSource(file_name)
        self._setSourceFormat('Neuralynx')
        
        # Format description for the NLX file:
        header_offset= 16*1024 # fixed for NLX files
        
        bytes_start_record= 2
        bytes_origin_id= 2
        bytes_videoRec_size= 2
        bytes_per_timestamp= 8
        bytes_per_bitfield= 4*400
        bytes_sncrc= 2
        bytes_per_xloc= 4
        bytes_per_yloc= 4
        bytes_per_angle= 4
        bytes_per_target= 4*50    
        
        record_size= None
        with open(file_name, 'rb') as f:
            while True:                
                line= f.readline()
                try:
                    line= line.decode('UTF-8')
                except:
                    break
                    
                if ""== line:
                    break
                if 'SamplingFrequency' in line:
                    self._setSamplingRate ( float(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if 'RecordSize' in line:
                    record_size= int(''.join(re.findall(r'\d+.\d+|\d+', line)))
                if 'Time Opened' in line:          
                    self._setDate(re.search(r'\d+/\d+/\d+', line).group())
                    self._setTime(re.search(r'\d+:\d+:\d+', line).group())
                if 'FileVersion' in line:
                    self._setFileVersion(line.split()[1])
            
            if not record_size:
                record_size= bytes_start_record+ \
                             bytes_origin_id+ \
                             bytes_videoRec_size+  \
                             bytes_per_timestamp+ \
                             bytes_per_bitfield+ \
                             bytes_sncrc+ \
                             bytes_per_xloc+ \
                             bytes_per_yloc+ \
                             bytes_per_angle+ \
                             bytes_per_target
            
            time_offset= bytes_start_record+ \
                             bytes_origin_id+ \
                             bytes_videoRec_size
            xloc_offset= time_offset+ \
                         bytes_per_timestamp+ \
                         bytes_per_bitfield+ \
                         bytes_sncrc
            yloc_offset= xloc_offset+bytes_per_xloc                        
            angle_offset= yloc_offset+bytes_per_xloc
            
            f.seek(0, 2)
            self._totalSamples= int((f.tell()- header_offset)/ record_size)
            spatial_data= np.zeros([self._totalSamples, 4])
            
            f.seek(header_offset)
            for i in np.arange(self._totalSamples):
                sample_bytes= np.fromfile(f, dtype= 'uint8', count= record_size)         
                spatial_data[i, 0]= int.from_bytes(sample_bytes[time_offset+ np.arange(bytes_per_timestamp)], byteorder= 'little', signed= False)
                spatial_data[i, 1]= int.from_bytes(sample_bytes[xloc_offset+ np.arange(bytes_per_xloc)], byteorder= 'little', signed= False)
                spatial_data[i, 2]= int.from_bytes(sample_bytes[yloc_offset+ np.arange(bytes_per_yloc)], byteorder= 'little', signed= False)
                spatial_data[i, 3]= int.from_bytes(sample_bytes[angle_offset+ np.arange(bytes_per_angle)], byteorder= 'little', signed= False)
            
            spatial_data[:, 0]/= 10**6
            spatial_data[:, 0]-= np.min(spatial_data[:, 0])
            self._timestamp= np.mean(np.diff(spatial_data[:, 0]))
            self._setSamplingRate(1/self._timestamp)
            self._setTime(spatial_data[:, 0])
            self._setPosX(spatial_data[:, 1]- np.min(spatial_data[:, 1]))
            self._setPosY(spatial_data[:, 2]- np.min(spatial_data[:, 2]))
            self._setDirection(spatial_data[:, 3])
#            self._setSpeed(spatial_data[:, 4]) # Neuralynx data does not have any speed information            
            self.smoothDirection()
            
    def smoothSpeed(self):
        self._setSpeed(smooth1D(self.getSpeed(), 'b', 5))
    def smoothDirection(self):
        cs= CircStat()
        cs.setTheta(self.getDirection())
        self._setDirection(cs.circSmooth(filttype= 'b', filtsize= 5))
            
    def calcAngVel(self, npoint= 5):
        theta= self.getDirection()
        angVel= np.zeros(theta.shape)
        N= theta.size
        L= npoint
        l= int(np.floor(L/2))
        cs= CircStat()
        for i in np.arange(l):
            y= cs.circRegroup(theta[:L-l+ i])
            angVel[i]= np.polyfit(np.arange(len(y)), y, 1)[0]
            
        for i in np.arange(l, N- l, 1):
            y= cs.circRegroup(theta[i-l:i+l+ 1])
            angVel[i]= np.polyfit(np.arange(len(y)), y, 1)[0]

        for i in np.arange(N- l, N):    
            y= cs.circRegroup(theta[i- l:])
            angVel[i]= np.polyfit(np.arange(len(y)), y, 1)[0]
        
        return angVel*self.getSamplingRate()
        
    def calcBorder(self, **kwargs):
        # define edges
        pixel= kwargs.get('pixel', 3)
        chopBound= kwargs.get('chopBound', 5)
        
        xedges= np.arange(0, np.ceil(np.max(self._posX)), pixel)
        yedges= np.arange(0, np.ceil(np.max(self._posY)), pixel)
            
        tmap, yedges, xedges= histogram2d(self._posY, self._posX, yedges, xedges)
        if abs(xedges.size- yedges.size)<= chopBound:
            tmap= chopEdges(tmap, min(tmap.shape), min(tmap.shape))[2]
        else:
            tmap= chopEdges(tmap, tmap.shape[1], tmap.shape[0])[2]
        
        ybin, xbin = tmap.shape
            
        border= np.zeros(tmap.shape)
        border[tmap> 0]= True
        border[tmap==0]= False
 
        for J in np.arange(ybin):
            for I in np.arange(xbin):
                if border[J, I] and (J== ybin-1 or J==  0 or I== xbin-1 or I== 0):
                    border[J, I]= False
 
        xborder= np.zeros(tmap.shape, dtype= bool)
        yborder= np.zeros(tmap.shape, dtype= bool)
        for J in np.arange(ybin):
            xborder[J, find(border[J, :], 1, 'first')-1]= True # 1 added/subed to the next pixel of the traversed arena as the border
            xborder[J, find(border[J, :], 1, 'last')+1]= True
        for I in np.arange(xbin):
            yborder[find(border[:, I], 1, 'first')-1, I]= True
            yborder[find(border[:, I], 1, 'last')+1, I]= True
               
#        self.border= border
        border= xborder | yborder
        self.tmap = tmap*self._timestamp
        
        distMat= np.zeros(border.shape)
        xx, yy= np.meshgrid(np.arange(xbin), np.arange(ybin))
        borderDist= np.zeros(self._time.size)
        
        xedges= np.arange(xbin)*pixel
        yedges= np.arange(ybin)*pixel
        xind= histogram(self._posX, xedges)[1]
        yind= histogram(self._posY, yedges)[1]
        
        for J in np.arange(ybin):
            for I in np.arange(xbin):
                tmp_dist= np.min(np.abs(xx[border]- xx[J, I])+ np.abs(yy[border]- yy[J, I]))
                if find(np.logical_and(xind==I, yind==J)).size:
                    borderDist[np.logical_and(xind==I, yind==J)]= tmp_dist
                distMat[J, I]=  tmp_dist
        return borderDist*pixel, xedges, yedges, distMat*pixel
        
    @staticmethod
    def skaggsInfo(firingRate, visitTime):
        firingRate[np.isnan(firingRate)]= 0
        Li= firingRate # Lambda
        L= np.sum(firingRate*visitTime)/ visitTime.sum()
        P= visitTime/visitTime.sum()
        return np.sum(P[Li>0]*(Li[Li>0]/L)*np.log2(Li[Li>0]/L))
        
    @staticmethod
    def spatialSparsity(firingRate, visitTime):
        firingRate[np.isnan(firingRate)]= 0
        Li= firingRate # Lambda
#        L= np.sum(firingRate*visitTime)/ visitTime.sum()
        P= visitTime/visitTime.sum()
        return np.sum(P*Li)**2/ np.sum(P*Li**2)
    
    def speed(self, ftimes, **kwargs):
#        if ftimes is None:
#            ftimes= self.getUnitStamp()
        _results= oDict()
        graphData= {}
        update= kwargs.get('update', True) # When update = True, it will use the 
                                            #results for statistics, if False, 
                                            #i.e. in Multiple Regression, it will ignore updating
        binsize= kwargs.get('binsize', 1)
        minSpeed, maxSpeed= kwargs.get('range', [0, 40])
        
        speed= self.getSpeed()
        maxSpeed= min(maxSpeed, np.ceil(speed.max()/binsize)*binsize)
        minSpeed= max(minSpeed, np.floor(speed.min()/binsize)*binsize)
        bins= np.arange(minSpeed, maxSpeed, binsize)

        vidCount= histogram(ftimes, self.getTime())[0]
        visitTime, speedInd= histogram(speed, bins)[0:2]
        visitTime= visitTime/self.getSamplingRate()
        
        rate = np.array([sum(vidCount[speedInd==i]) for i in range(len(bins))])/ visitTime
        rate[np.isnan(rate)]= 0 
        
        _results['speedSkaggs']= self.skaggsInfo(rate,  visitTime)  
        
        rate= rate[visitTime> 1]
        bins= bins[visitTime> 1]
    
        fit_result= linfit(bins, rate)
        
        _results['speedPearsR']= fit_result['Pearson R']
        _results['speedPearsP']= fit_result['Pearson P']
        graphData['bins']= bins
        graphData['rate']= rate
        graphData['fitRate']= fit_result['yfit']
        
        if update:
            self.updateResult(_results)
        return graphData
    
    def angularVelocity(self, ftimes, **kwargs):
#        if ftimes is None:
#            ftimes= self.getUnitStamp()
        _results= oDict()
        graphData= {}
        update= kwargs.get('update', True) # When update = True, it will use the 
                                            #results for statistics, if False, 
                                            #i.e. in Multiple Regression, it will ignore updating
        binsize= kwargs.get('binsize', 10)
        minVel, maxVel= kwargs.get('range', [-100, 100])
        cutoff= kwargs.get('cutoff', 10)
        
        angVel= self.getAngVel()
        
        maxVel= min(maxVel, np.ceil(angVel.max()/binsize)*binsize)
        minVel= max(minVel, np.floor(angVel.min()/binsize)*binsize)
        bins= np.arange(minVel, maxVel, binsize)

        vidCount= histogram(ftimes, self.getTime())[0]
        visitTime, velInd= histogram(angVel, bins)[0:2]
        visitTime= visitTime/self.getSamplingRate()
        
        rate = np.array([sum(vidCount[velInd==i]) for i in range(len(bins))])/ visitTime
        rate[np.isnan(rate)]= 0
        
        _results['speedSkaggs']= self.skaggsInfo(rate,  visitTime)  
        
        rate= rate[visitTime> 1]
        bins= bins[visitTime> 1]
        
    
        fit_result= linfit(bins[bins<= -cutoff], rate[bins<= -cutoff])
        
        _results['angVelLeftPearsR']= fit_result['Pearson R']
        _results['angVelLeftPearsP']= fit_result['Pearson P']
        graphData['leftBins']= bins[bins<= -cutoff]
        graphData['leftRate']= rate[bins<= -cutoff]
        graphData['leftFitRate']= fit_result['yfit']
        
        fit_result= linfit(bins[bins>= cutoff], rate[bins>= cutoff])
        
        _results['angVelRightPearsR']= fit_result['Pearson R']
        _results['angVelRightPearsP']= fit_result['Pearson P']
        graphData['rightBins']= bins[bins>= cutoff]
        graphData['rightRate']= rate[bins>= cutoff]
        graphData['rightFitRate']= fit_result['yfit']
        
        if update:
            self.updateResult(_results)
        return graphData
        
    def place(self, ftimes, **kwargs):
#        if ftimes is None:
#            ftimes= self.getUnitStamp()
        _results= oDict()
        graphData= {}
        update= kwargs.get('update', True)
        pixel= kwargs.get('pixel', 3)
        chopBound= kwargs.get('chopBound', 5)
        filttype, filtsize= kwargs.get('filter', ['b', 5])
        lim= kwargs.get('range', [0, self.getDuration()])
        brAdjust= kwargs.get('brAdjust', True)
#        thresh= kwargs.get('fieldThresh', 0.2)
        
#        xedges= np.arange(0, np.ceil(np.max(self._posX)), pixel)
#        yedges= np.arange(0, np.ceil(np.max(self._posY)), pixel)
        xedges= self._xbound
        yedges= self._ybound

        spikeLoc= self.getEventLoc(ftimes, **kwargs)[1]
        posX= self._posX[np.logical_and(self.getTime()>= lim[0], self.getTime()<= lim[1])] 
        posY= self._posY[np.logical_and(self.getTime()>= lim[0], self.getTime()<= lim[1])]
        
        tmap, yedges, xedges= histogram2d(posY, posX, yedges, xedges)
        
        if tmap.shape[0] != tmap.shape[1] & np.abs(tmap.shape[0]- tmap.shape[1])<= chopBound:
            tmap= chopEdges(tmap, min(tmap.shape), min(tmap.shape))[2]
        tmap/= self.getSamplingRate()
        
        ybin, xbin= tmap.shape
        xedges= np.arange(xbin)*pixel
        yedges= np.arange(ybin)*pixel
        
        spikeCount= histogram2d(spikeLoc[1], spikeLoc[0], yedges, xedges)[0]
        fmap= np.divide(spikeCount, tmap, out= np.zeros_like(spikeCount), where= tmap!= 0)
        
        if brAdjust:
            nfmap= fmap/ fmap.max()
            if np.sum(np.logical_and(nfmap>= 0.2, tmap!= 0))>= 0.8*nfmap[tmap!= 0].flatten().shape[0]:
                back_rate= np.mean(fmap[np.logical_and(nfmap>= 0.2, nfmap< 0.4)])
                fmap -= back_rate
                fmap[fmap<0]= 0
#        pfield= self.placeField(fmap, thresh)
        
        smoothMap= smooth2d(fmap, filttype, filtsize)
        
        if update:
            _results['spatialSkaggs']= self.skaggsInfo(fmap, tmap)
            _results['spatialSparsity']= self.spatialSparsity(fmap, tmap)
            _results['spatialCoherence']= np.corrcoef(fmap[tmap != 0].flatten(), smoothMap[tmap != 0].flatten())[0, 1]
            self.updateResult(_results)
            
        smoothMap[tmap==0]= None

        graphData['posX']= posX
        graphData['posY']= posY                
        graphData['fmap']= fmap
        graphData['smoothMap']= smoothMap
        graphData['tmap']= tmap
        graphData['xedges']= xedges
        graphData['yedges']= yedges
        graphData['spikeLoc']= spikeLoc
        
        return graphData
        
    def locTimeLapse(self, ftimes, **kwargs):
        #### Breaking down the spike plot for firing evolution
        # 0-1min,  0-2min, 0-4min, 0-8min
        graphData= oDict()
        pixel= kwargs.get('pixel', 3)
        chopBound= kwargs.get('chopBound', 5)
        filter= kwargs.get('filter', ['b', 5])
        brAdjust= kwargs.get('brAdjust', True)
        
        lim= [0, 1*60]
        graphData['0To1min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)

        lim= [0, 2*60]
        graphData['0To2min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
        lim= [0, 4*60]
        graphData['0To4min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
        
        lim= [0, 8*60]
        graphData['0To8min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
        
        # 0-1min, 1-2min,  2-4min, 4-8min        
        lim= [1*60, 2*60]        
        graphData['1To2min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
        
        lim= [2*60, 4*60]
        graphData['2To4min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
        
        lim= [4*60, 8*60]
        graphData['4To8min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
        
        ## 0-16min, 8-16min 0-20min, 16-20min
        
        if self.getDuration()> 8*60:
            if self.getDuration()> 16*60:
                lim= [0, 16*60]
                graphData['0To16min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
                lim= [8*60, 16*60]
                graphData['8To16min']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
                lim= [0, self.getDuration()]                
                graphData['0ToEnd']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
                lim= [16*60, self.getDuration()]
                graphData['16ToEnd']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
            else:
                lim= [0, self.getDuration()]
                graphData['0ToEnd']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
                lim= [8*60, self.getDuration()]
                graphData['8ToEnd']= self.place(ftimes, range= lim, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
                
            return graphData
        
    def hdRate(self, ftimes, **kwargs):
        _results= oDict()
        graphData= {}
        update= kwargs.get('update', True)
        binsize= kwargs.get('binsize', 5) # in degrees
        filttype, filtsize= kwargs.get('filter', ['b', 5])
        lim= kwargs.get('range', [0, self.getDuration()])

        bins= np.arange(0, 360, binsize)
        
        spike_hd= self.getEventLoc(ftimes, **kwargs)[2]
        direction= self.getDirection()[np.logical_and(self.getTime()>= lim[0], self.getTime()<= lim[1])]
        
        tcount, ind, bins = histogram(direction, bins)
        
        tcount= tcount/ self.getSamplingRate()
                
        spikeCount= histogram(spike_hd, bins)[0].astype(tcount.dtype)

        hdRate= np.divide(spikeCount, tcount, out= np.zeros_like(spikeCount), where= tcount!= 0, casting= 'unsafe')
        
        smoothRate= smooth1D(hdRate, filttype, filtsize)
        
        if update:
            _results['hdSkaggs']= self.skaggsInfo(hdRate, tcount)
            cs= CircStat(rho= smoothRate, theta= bins)
            results= cs.calcStat()
            _results['hdRaylZ']= results['RaylZ']
            _results['hdRaylP']= results['RaylP']
            _results['hdVonMissesK']= results['vonMissesK']
            _results['hdMean']= results['meanTheta']
            _results['hdMeanRate']= results['meanRho']
            _results['hdResVect']= results['resultant']
            
            binInterp= np.arange(360)
            rateInterp= np.interp(binInterp, bins, hdRate)
            
            _results['hdPeakRate']= np.amax(rateInterp)
            _results['hdPeak']= binInterp[np.argmax(rateInterp)]
            
            half_max= np.amin(rateInterp)+ (np.amax(rateInterp)- np.amin(rateInterp))/2
            d = np.sign(half_max - rateInterp[0:-1]) - np.sign(half_max - rateInterp[1:])
            left_idx= find(d > 0)[0]
            right_idx= find(d < 0)[-1]
            _results['hdHalfWidth']= None if (not left_idx or not right_idx or left_idx> right_idx) \
                                        else binInterp[right_idx]- binInterp[left_idx]
            
            pixel= kwargs.get('pixel',  3)
            placeData= self.place(ftimes, pixel= pixel)
            fmap= placeData['smoothMap']
            fmap[np.isnan(fmap)]= 0
            hdPred= np.zeros(bins.size) 
            for i, b in enumerate(bins):
                hdInd= np.logical_and(direction>= b, direction< b+ binsize)
                tmap= histogram2d(self.getY()[hdInd], self.getX()[hdInd], placeData['yedges'], placeData['xedges'])[0]
                tmap/= self.getSamplingRate()
                hdPred[i]= np.sum(fmap*tmap)/ tmap.sum()                             
            
            graphData['hdPred']= smooth1D(hdPred, 'b', 5)
            self.updateResult(_results)            
            
        graphData['hd']= direction
        graphData['hdRate']= hdRate
        graphData['smoothRate']= smoothRate
        graphData['tcount']= tcount
        graphData['bins']= bins
        graphData['spike_hd']= spike_hd
        
        cs= CircStat()
        cs.setTheta(spike_hd)
        graphData['scatter_radius'], graphData['scatter_bins']= cs.circScatter(bins= 2, step= 0.05)
        
        
        return graphData
        
    def hdRateCCW(self, ftimes, **kwargs):
        _results= oDict()
        graphData= {}
        update= kwargs.get('update', True)
        binsize= kwargs.get('binsize', 5) # in degrees
        filttype, filtsize= kwargs.get('filter', ['b', 5])
        lim= kwargs.get('range', [0, self.getDuration()])
        thresh= kwargs.get('thresh', 30)

        edges= np.arange(0, 360, binsize)
        
        spikeInd, spikeLoc, spike_hd= self.getEventLoc(ftimes, **kwargs)
        vidInd= np.logical_and(self.getTime()>= lim[0], self.getTime()<= lim[1])
        direction= self.getDirection()[vidInd]
        
        ccwSpike_hd= spike_hd[self.getAngVel()[spikeInd]< -thresh]
        cwSpike_hd= spike_hd[self.getAngVel()[spikeInd]> thresh]

        ccw_dir= direction[self.getAngVel()[vidInd]< -thresh]
        cw_dir= direction[self.getAngVel()[vidInd]> thresh]
        
        
        binInterp= np.arange(360)
        
        tcount, ind, bins = histogram(cw_dir, edges)
        tcount= tcount/ self.getSamplingRate()
        spikeCount= histogram(cwSpike_hd, edges)[0].astype(tcount.dtype)
        cwRate= np.divide(spikeCount, tcount, out= np.zeros_like(spikeCount), where= tcount!= 0, casting= 'unsafe')
        cwRate= np.interp(binInterp, bins, smooth1D(cwRate, filttype, filtsize))
        
        tcount, ind, bins = histogram(ccw_dir, edges)
        tcount= tcount/ self.getSamplingRate()
        spikeCount= histogram(ccwSpike_hd, edges)[0].astype(tcount.dtype)
        ccwRate= np.divide(spikeCount, tcount, out= np.zeros_like(spikeCount), where= tcount!= 0, casting= 'unsafe')                        
        ccwRate= np.interp(binInterp, bins, smooth1D(ccwRate, filttype, filtsize))        
        
        if update:
            _results['hdDelta']= binInterp[np.argmax(ccwRate)]- binInterp[np.argmax(cwRate)]
            _results['hdPeakCW']= np.argmax(cwRate)
            _results['hdPeakCCW']= np.argmax(ccwRate)
            _results['hdPeakRateCW']= np.amax(cwRate)
            _results['hdPeakRateCCW']= np.amax(ccwRate)
            self.updateResult(_results)   

        graphData['bins']= binInterp
        graphData['hdRateCW']= cwRate
        graphData['hdRateCCW']= ccwRate
        
        return graphData
        
    def hdTimeLapse(self, ftimes):
        #### Breaking down the spike plot for firing evolution
        # 0-1min,  0-2min, 0-4min, 0-8min
        graphData= oDict()
        lim= [0, 1*60]
        graphData['0To1min']= self.hdRate(ftimes, range= lim, update= False)

        lim= [0, 2*60]
        graphData['0To2min']= self.hdRate(ftimes, range= lim, update= False)
                
        lim= [0, 4*60]
        graphData['0To4min']= self.hdRate(ftimes, range= lim, update= False)
        
        lim= [0, 8*60]
        graphData['0To8min']= self.hdRate(ftimes, range= lim, update= False)
        
        # 0-1min, 1-2min,  2-4min, 4-8min        
        lim= [1*60, 2*60]        
        graphData['1To2min']= self.hdRate(ftimes, range= lim, update= False)
        
        lim= [2*60, 4*60]
        graphData['2To4min']= self.hdRate(ftimes, range= lim, update= False)
        
        lim= [4*60, 8*60]
        graphData['4To8min']= self.hdRate(ftimes, range= lim, update= False)
        
        ## 0-16min, 8-16min 0-20min, 16-20min
        
        if self.getDuration()> 8*60:
            if self.getDuration()> 16*60:
                lim= [0, 16*60]
                graphData['0To16min']= self.hdRate(ftimes, range= lim, update= False)
                
                lim= [8*60, 16*60]
                graphData['8To16min']= self.hdRate(ftimes, range= lim, update= False)
                
                lim= [0, self.getDuration()]                
                graphData['0ToEnd']= self.hdRate(ftimes, range= lim, update= False)
                
                lim= [16*60, self.getDuration()]
                graphData['16ToEnd']= self.hdRate(ftimes, range= lim, update= False)
                
            else:
                lim= [0, self.getDuration()]
                graphData['0ToEnd']= self.hdRate(ftimes, range= lim, update= False)
                
                lim= [8*60, self.getDuration()]
                graphData['8ToEnd']= self.hdRate(ftimes, range= lim, update= False)
        
            return graphData
            
    def hdShuffle(self, ftimes, **kwargs): 
        _results= oDict()
        graphData= {}
        nshuff= kwargs.get('nshuff', 500)
        limit= kwargs.get('limit', 0)
        bins= kwargs.get('bins', 100)
        # limit= 0 implies enirely random shuffle, limit= 'x' implies nshuff number of shuffles in the range [-x x]
        dur= self.getTime()[-1]
        shift = nprand.uniform(low= limit- dur, high= dur- limit, size= nshuff)
        raylZ= np.zeros((nshuff,))
        for i in np.arange(nshuff):
            shift_ftimes= ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes> dur] -= dur
            shift_ftimes[shift_ftimes< 0] += dur
            
            hdData= self.hdRate(shift_ftimes, update= False)
            cs= CircStat(rho= hdData['smoothRate'], theta= hdData['bins'])
            results= cs.calcStat()
            raylZ[i]= results['RaylZ']

        graphData['raylZ']= raylZ
        hdData= self.hdRate(ftimes, update= False)
        cs.setRho(hdData['smoothRate'])
        results= cs.calcStat()
        graphData['refRaylZ']= results['RaylZ']
        
        graphData['raylZCount'], ind,  graphData['raylZEdges']= histogram(raylZ, bins= bins)
        graphData['per95']= np.percentile(raylZ, 95)

        _results['hdShuffPer95']= np.percentile(raylZ, 95)
        self.updateResult(_results)

        return graphData
        
    def hdShift(self, ftimes, shiftInd= np.arange(-10, 11)):
        _results= oDict()
        graphData= {}
        shift= shiftInd/self.getSamplingRate()
        shiftlen= shift.size
        dur= self.getTime()[-1]
        delta= np.zeros((shiftlen,))
        skaggs= np.zeros((shiftlen,))
        peakRate= np.zeros((shiftlen,))
        
        for i in np.arange(shiftlen):
            shift_ftimes= ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes> dur] -= dur
            shift_ftimes[shift_ftimes< 0] += dur
            
            hdData= self.hdRateCCW(shift_ftimes, update= False)
            delta[i]= hdData['bins'][np.argmax(hdData['hdRateCCW'])]- hdData['bins'][np.argmax(hdData['hdRateCW'])]
            hdData= self.hdRate(shift_ftimes, update= False)
            peakRate[i]= np.amax(hdData['smoothRate'])
            skaggs[i]= self.skaggsInfo(hdData['hdRate'], hdData['tcount'])

        graphData['delta']= delta
        graphData['skaggs']= skaggs
        graphData['peakRate']= peakRate
        graphData['shiftTime']= shift*1000 # changing to milisecond
            
        # Find out the optimum skaggs location
        shiftUpsamp= np.arange(shift[0], shift[-1], np.mean(np.diff(shift))/10)
        skaggsUpsamp= np.interp(shiftUpsamp, shift, skaggs)
        peakRateUpsamp= np.interp(shiftUpsamp, shift, peakRate)
        
        dfit_result= linfit(shift, delta)
        deltaFit= dfit_result['yfit']
        sortInd= np.argsort(deltaFit)
        _results['hdATI']= np.interp(0, deltaFit[sortInd], shift[sortInd])*1000 if dfit_result['Pearson R']>= 0.85 else None
        
        graphData['deltaFit']= deltaFit
        imax= sg.argrelmax(skaggsUpsamp)[0]
        maxloc= find(skaggsUpsamp[imax]== skaggsUpsamp.max())
        _results['hdOptShiftSkaggs']= np.nan if maxloc.size!=1 else \
                    (np.nan if imax[maxloc]==0 or imax[maxloc]== skaggsUpsamp.size else shiftUpsamp[imax[maxloc]][0]*1000) # in milisecond

        imax= sg.argrelmax(peakRateUpsamp)[0]
        maxloc= find(peakRateUpsamp[imax]== peakRateUpsamp.max())
        _results['hdOptShiftPeakRate']= np.nan if maxloc.size!=1 else \
                    (np.nan if imax[maxloc]==0 or imax[maxloc]== peakRateUpsamp.size else shiftUpsamp[imax[maxloc]][0]*1000) # in milisecond
        self.updateResult(_results)
        
        return graphData
    
#    @staticmethod
#    def placeMap(pmap, thresh= 0.2):
#        
#        def alongColumn(pfield, ptag, J, I):
#            Ji= J 
#            Ii= I
#            rows=[]
#            while J+1< ptag.shape[0]:
#                if not pfield[J+1, I] or ptag[J+1, I]:
#                    break              
#                else:
#                    ptag[J+1, I]= ptag[J, I]
#                    rows.append(J+1)
#                    J+=1                    
#            J= Ji
#            while J-1>0:
#                if not pfield[J-1, I] or ptag[J-1, I]:
#                    break
#                else:
#                    ptag[J-1, I]= ptag[J, I]
#                    rows.append(J-1)
#                    J-=1
##            rows= find(ptag[:, I]== ptag[Ji, Ii])
#            for J in rows:
#                if J!= Ji:
#                    ptag= alongRows(pfield, ptag, J, Ii)
#            return ptag
#                
#        def alongRows(pfield, ptag, J, I):
#            Ji= J
#            Ii= I
#            columns=[]
#            while I+1<= ptag.shape[1]:
#                if not pfield[J, I+1] or ptag[J, I+1]: 
#                    break
#                else:
#                    ptag[J, I+1]= ptag[J, I]
#                    columns.append(I+1)
#                    I+=1
#            I= Ii
#            while I-1>=0:
#                if not pfield[J, I-1] or ptag[J, I-1]:
#                    break
#                else:
#                    ptag[J, I-1]= ptag[J, I]
#                    columns.append(I-1)
#                I-=1
##            columns= find(ptag[J, ]== ptag[Ji, Ii])
#            for I in columns:
#                if I!= Ii:
#                    ptag=  alongColumn(pfield, ptag, J, I)            
#            return ptag
#        # Finding the place map firing field:
#        # Rules: 1. spikes in bin 2. The bin shares at least a side with other bins which contain spikes
#        pmap= pmap/pmap.max()
#        pmap= pmap> thresh
#        pfield= np.zeros(np.add(pmap.shape,2))
#        pfield[1:-1, 1:-1]= pmap
#        
#        pfield[1:-1, 1:-1]= np.logical_and(pmap, np.logical_or(np.logical_or(pfield[0:-2, 1:-1], pfield[2:, 1:-1]), \
#                            np.logical_or(pfield[1:-1, 0:-2], pfield[1:-1, 2:]))) # shifted and tested for neighboring pixel spike occupation
#        # tags start at 2; Will be renumbered based on sizes of the fields
#        group= 2
#        ptag= np.zeros(pfield.shape, dtype= int)
#    
#        J, I= find2d(pfield, 1)
#        J= J[0]
#        I= I[0]
#        ptag[J, I]= group
#        
#        ptag= alongColumn(pfield, ptag, J, I)
        
    def getEventLoc(self, ftimes, **kwargs):
        time= self.getTime()
        lim= kwargs.get('range', [0, time.max()])
        vidInd= histogram(ftimes[np.logical_and(ftimes >= lim[0], ftimes< lim[1])], time)[1]
        return vidInd[vidInd!=0], [self._posX[vidInd[vidInd!=0]], self._posY[vidInd[vidInd!=0]]], self._direction[vidInd[vidInd!=0]]
                
    def locShuffle(self, ftimes, **kwargs):
        _results= oDict()
        graphData= {}
        
        nshuff= kwargs.get('nshuff', 500)
        limit= kwargs.get('limit', 0)
        bins= kwargs.get('bins', 100)
        brAdjust= kwargs.get('brAdjust', False)
        pixel= kwargs.get('pixel', 3)
        chopBound= kwargs.get('chopBound', 5)
        filter= kwargs.get('filter', ['b', 5])
        
        # limit= 0 implies enirely random shuffle, limit= 'x' implies nshuff number of shuffles in the range [-x x]
        dur= self.getTime()[-1]
        shift = nprand.uniform(low= limit- dur, high= dur- limit, size= nshuff)
        skaggs= np.zeros((nshuff,))
        sparsity= np.zeros((nshuff,))
        coherence= np.zeros((nshuff,))
        for i in np.arange(nshuff):
            shift_ftimes= ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes> dur] -= dur
            shift_ftimes[shift_ftimes< 0] += dur
            
            placeData= self.place(shift_ftimes, filter= filter, \
                          chopBound= chopBound, pixel= pixel, brAdjust= brAdjust, update= False)
            skaggs[i]= self.skaggsInfo(placeData['fmap'], placeData['tmap'])
            sparsity[i]= self.spatialSparsity (placeData['fmap'], placeData['tmap'])
            coherence[i]= np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]

        graphData['skaggs']= skaggs
        graphData['coherence']= coherence
        graphData['sparsity']= sparsity
        
        placeData= self.place(ftimes, pixel= pixel, filter= filter, brAdjust= brAdjust,\
                              chopBound= chopBound, update= False)
        graphData['refSkaggs']= self.skaggsInfo(placeData['fmap'], placeData['tmap'])
        graphData['refSparsity']= self.spatialSparsity(placeData['fmap'], placeData['tmap'])
        graphData['refCoherence']= np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]
        
        graphData['skaggsCount'], graphData['skaggsEdges']= np.histogram(skaggs, bins= bins)
        graphData['skaggs95']= np.percentile(skaggs, 95)

        graphData['sparsityCount'], graphData['sparsityEdges']= np.histogram(sparsity, bins= bins)
        graphData['sparsity05']= np.percentile(sparsity, 5)
        
        graphData['coherenceCount'], graphData['coherenceEdges']= np.histogram(coherence, bins= bins)
        graphData['coherence95']= np.percentile(coherence, 95)
        
        _results['locSkaggs95']= np.percentile(skaggs, 95)
        _results['locSparsity05']= np.percentile(sparsity, 95)
        _results['locCoherence95']= np.percentile(coherence, 95)
        
        self.updateResult(_results)

        return graphData
        
    def locShift(self, ftimes, shiftInd= np.arange(-10, 11), **kwargs):
        _results= oDict()
        graphData= {}
        
        brAdjust= kwargs.get('brAdjust', False)
        pixel= kwargs.get('pixel', 3)
        chopBound= kwargs.get('chopBound', 5)
        filter= kwargs.get('filter', ['b', 5])
        
        # limit= 0 implies enirely random shuffle, limit= 'x' implies nshuff number of shuffles in the range [-x x]
        shift= shiftInd/self.getSamplingRate()
        shiftlen= shift.size
        dur= self.getTime()[-1]
        skaggs= np.zeros((shiftlen,))
        sparsity= np.zeros((shiftlen,))
        coherence=  np.zeros((shiftlen,))
        
        for i in np.arange(shiftlen):
            shift_ftimes= ftimes+ shift[i]
            # Wrapping up the time
            shift_ftimes[shift_ftimes> dur] -= dur
            shift_ftimes[shift_ftimes< 0] += dur
            
            placeData= self.place(shift_ftimes, pixel= pixel, filter= filter, \
                                  brAdjust= brAdjust,chopBound= chopBound, update= False)
            skaggs[i]= self.skaggsInfo(placeData['fmap'], placeData['tmap'])
            sparsity[i]= self.spatialSparsity (placeData['fmap'], placeData['tmap'])
            coherence[i]= np.corrcoef(placeData['fmap'][placeData['tmap'] != 0].flatten(), \
                            placeData['smoothMap'][placeData['tmap'] != 0].flatten())[0, 1]            

        graphData['skaggs']= skaggs 
        graphData['sparsity']= sparsity
        graphData['coherence']= coherence
        
        graphData['shiftTime']= shift
            
        # Find out the optimum skaggs location
        shiftUpsamp= np.arange(shift[0], shift[-1], np.mean(np.diff(shift))/4)
        skaggsUpsamp= np.interp(shiftUpsamp, shift, skaggs)
        sparsityUpsamp= np.interp(shiftUpsamp, shift, sparsity)
        coherenceUpsamp= np.interp(shiftUpsamp, shift, coherence)
        
        imax= sg.argrelmax(skaggsUpsamp)[0]
        maxloc= find(skaggsUpsamp[imax]== skaggsUpsamp.max())
        _results['locOptShiftSkaggs']= np.nan if maxloc.size!=1 else (np.nan if imax[maxloc]==0 or imax[maxloc]== skaggsUpsamp.size else shiftUpsamp[imax[maxloc]])
        
        imin= sg.argrelmin(sparsityUpsamp)[0]
        minloc= find(sparsityUpsamp[imin]== sparsityUpsamp.min())
        _results['locOptShiftSparsity']= np.nan if minloc.size!=1 else (np.nan if imin[minloc]==0 or imin[minloc]== sparsityUpsamp.size else shiftUpsamp[imin[minloc]])        

        imax= sg.argrelmax(coherenceUpsamp)[0]
        maxloc= find(coherenceUpsamp[imax]== coherenceUpsamp.max())
        _results['locOptShiftCoherence']= np.nan if maxloc.size!=1 else (np.nan if imax[maxloc]==0 or imax[maxloc]== coherenceUpsamp.size else shiftUpsamp[imax[maxloc]])                
        
        self.updateResult(_results)
        
        return graphData
        
    def locAutoCorr(self, ftimes, **kwargs):
        graphData= {}
        
        minPixel= kwargs.get('minPixel', 100)
        pixel= kwargs.get('pixel', 3)
        
        if 'update' in kwargs.keys():
            del kwargs['update']
        placeData= self.place(ftimes, update= False, **kwargs)
        
        fmap= placeData['smoothMap']
        fmap[np.isnan(fmap)]= 0
        leny, lenx= fmap.shape
        
        xshift= np.arange(-(lenx-1), lenx)
        yshift= np.arange(-(leny-1), leny)
        
        corrMap= np.zeros((yshift.size, xshift.size))
        
        for J, ysh  in enumerate(yshift):
            for I, xsh in enumerate(xshift):
                if ysh >= 0:
                    map1YInd= np.arange(ysh, leny)
                    map2YInd=  np.arange(leny- ysh)
                elif ysh < 0:
                    map1YInd= np.arange(leny+ ysh)
                    map2YInd= np.arange(-ysh, leny)

                if xsh >= 0:
                    map1XInd= np.arange(xsh, lenx)
                    map2XInd=  np.arange(lenx- xsh)
                elif xsh < 0:
                    map1XInd= np.arange(lenx+ xsh)
                    map2XInd= np.arange(-xsh, lenx)                    
                map1=  fmap[np.meshgrid(map1YInd, map1XInd)]
                map2=  fmap[np.meshgrid(map2YInd, map2XInd)]
                if map1.size< minPixel:
                    corrMap[J, I]= -1
                else:          
                    corrMap[J, I]= corrCoeff(map1, map2)
                    
        graphData['corrMap']= corrMap
        graphData['xshift']= xshift*pixel
        graphData['yshift']= yshift*pixel
            
        return graphData
            
    def locRotCorr(self, ftimes, **kwargs):
        graphData= {}
        
        binsize= kwargs.get('binsize', 3) #degrees
#        filttype, filtsize= kwargs.get('filter', ['b', 3])

        bins= np.arange(0, 360, binsize)
        placeData= self.place(ftimes, update= False, **kwargs)
        
        fmap= placeData['smoothMap']
        fmap[np.isnan(fmap)]= 0
        
        rotCorr= [ corrCoeff(rot2d(fmap, theta), fmap) for k, theta in enumerate(bins)]
                
        graphData['rotAngle']= bins
        graphData['rotCorr']= rotCorr
            
        return graphData
        
    def border(self, ftimes, **kwargs ):
        _results= oDict()
        graphData= {}
        
        dist, xedges, yedges, distMat= self.getBorder()
        pixel= np.diff(xedges).mean()
        
        update= kwargs.get('update', True)
        thresh= kwargs.get('thresh', 0.2)
        cbinsize= kwargs.get('cbinsize', 5) # Circular binsize in degrees
        lim= kwargs.get('range', [0, self.getDuration()])
        
        steps= kwargs.get('nstep', 5)

        distBins= np.arange(dist.min(), dist.max()+ pixel, pixel)
        
        if 'update' in kwargs.keys():
            del kwargs['update']
            
        placeData= self.place(ftimes, range= lim, update= False, **kwargs)        
        fmap= placeData['smoothMap']
        
        xind= []
        yind= []
        if placeData['xedges'].max()< xedges.max():
            xind= xedges<= placeData['xedges'].max()
            xedges= xedges[xind]
        if placeData['yedges'].max()< yedges.max():
            yind= yedges<= placeData['yedges'].max()
            yedges= yedges[xind]
        
        if xind.any():
            distMat= distMat[:, xind]
        if yind.any():
            distMat= distMat[yind, :]
        
        nanInd= np.isnan(fmap)
        fmap[nanInd]= 0
        
        smoothRate= np.zeros(distBins.shape) # Calculated from smooth FR map not by smoothing from raw rate
        for i, edge in enumerate(distBins):
            edge_ind= distMat== edge
            if edge_ind.any() and np.logical_and(np.logical_not(nanInd), distMat== edge).any():
                smoothRate[i]= fmap[np.logical_and(np.logical_not(nanInd), distMat== edge)].mean()
#        smoothRate= smooth1D(smoothRate, filttype, filtsize)
        
        fmap/= fmap.max()
        
        tcount= histogram(dist, distBins)[0]
        
        tcount= tcount/ self.getSamplingRate()
        
        spikeDist= dist[self.getEventLoc(ftimes)[0]]        
        spikeCount= histogram(spikeDist, distBins)[0].astype(tcount.dtype)

        distRate= np.divide(spikeCount, tcount, out= np.zeros_like(spikeCount),\
                            where= tcount!= 0, casting= 'unsafe') # for skaggs only
        
        pixelCount= histogram(distMat[np.logical_not(nanInd)], distBins)[0]
        distCount= np.divide(histogram(distMat[fmap>= thresh], distBins)[0], pixelCount, \
                             out=np.zeros_like(distBins), where=pixelCount!=0, casting= 'unsafe')
        
        circBins= np.arange(0, 360, cbinsize)
        
        X, Y= np.meshgrid(xedges, np.flipud(yedges))
        X= X- xedges[-1]/2
        Y= Y- yedges[-1]/2
        angDist= np.arctan2(Y, X)* 180/np.pi
        angDist[angDist< 0]+= 360
        
        meanDist= distMat[fmap>=thresh].mean()
        
        cs= CircStat()
        cs.setTheta(angDist[np.logical_and(distMat<= meanDist, fmap>= thresh)])
        angDistCount= cs.circHistogram(circBins)[0]
        
        # Circular linear map
        circLinMap= np.zeros((distBins.size, circBins.size))
        
        for i, edge in enumerate(distBins):
            cs.setTheta(angDist[np.logical_and(distMat== edge, fmap>= thresh)])
            circLinMap[i, :]= cs.circHistogram(circBins)[0]     
  
        perSteps= np.arange(0, 1, 1/steps)
        perDist= np.zeros(steps)
        
        for i in np.arange(steps):
            perDist[i]= distMat[np.logical_and(np.logical_not(nanInd), \
                        np.logical_and(fmap>= perSteps[i], fmap< perSteps[i]+ 1/steps))].mean()
        if update:
            _results['borderSkaggs']= self.skaggsInfo(distRate, tcount)
            
            angDistExt= np.append(angDistCount, angDistCount)
            
            segsize= findChunk(angDistExt> 0)[0]
            _results['borderAngExt']= max(segsize)*cbinsize
        
            cBinsInterp= np.arange(0, 360, 0.1)
            dBinsInterp= np.arange(0, distBins[-1]+pixel, 0.1)            
            graphData['cBinsInterp']= cBinsInterp
            graphData['dBinsInterp']= dBinsInterp
            graphData['circLinMap']= sc.interpolate.interp2d(circBins, distBins, circLinMap, kind='cubic')(cBinsInterp, dBinsInterp)
            
            self.updateResult(_results)

        graphData['distBins']= distBins
        graphData['distCount']= distCount
        graphData['circBins']= circBins
        graphData['angDistCount']= angDistCount
        graphData['distRate']= distRate
        graphData['smoothRate']= smoothRate
        graphData['perSteps']= perSteps*100
        graphData['perDist']= perDist
        
        return graphData
        
    def gradient(self, ftimes, **kwargs):
        _results= oDict()
        graphData= {}
        
        alim= kwargs.get('alim', 0.25)
        blim= kwargs.get('blim', 0.25)
        clim= kwargs.get('clim', 0.5)
        
        graphData= self.border(ftimes, **kwargs)
        
        x= graphData['distBins']
        y= graphData['smoothRate']
        x= x[np.isfinite(y)]
        y= y[np.isfinite(y)]
        y= np.log(y, out= np.zeros_like(y), where= y!= 0, casting= 'unsafe')
        ai= y.max()
        y0= y[x== 0]
        bi= ai- y0
        
        d_half= x.mean()
        for i, dist in enumerate(x):
            if i< x.size-1 and (y0+ bi/2)> y[i] and (y0+ bi/2)<= y[i+1]:
                d_half= x[i:i+2].mean()
                
        ci= np.log(2)/d_half
        
        def fit_func(x, a, b, c):
            return a- b*np.exp(-c*x)
                
        popt, pcov = curve_fit(fit_func, x, y, \
                                p0= [ai, bi, ci], \
                                bounds= ([(1- alim)*ai, (1- blim)*bi, (1- clim)*ci], \
                                [(1+ alim)*ai, (1+ blim)*bi, (1+ clim)*ci]), \
                                max_nfev= 100000)
        a, b, c= popt 
        
        y_fit= fit_func(x, *popt)
        
        
        gof= residualStat(y, y_fit, 3)
        rateFit= np.exp(y_fit)
        
        graphData['distBins']= x
#        graphData['smoothRate']= y
        graphData['rateFit']=  rateFit
        graphData['diffRate']= b*c*np.multiply(rateFit, np.exp(-c*x))
        
        _results['grad Pearse R']= gof['Pearson R']
        _results['grad Pearse P']= gof['Pearson P']
        _results['grad adj Rsq']= gof['adj Rsq']
        _results['grad Max Growth Rate']= c*np.exp(a-1)
        _results['grad Inflect Dist']= np.log(b)/c
        
        self.updateResult(_results)
        return graphData

    def grid(self, ftimes, **kwargs):
        _results= oDict()
        tol= kwargs.get('angtol', 2)
        binsize= kwargs.get('binsize', 3)
        bins= np.arange(0, 360, binsize)

        graphData= self.locAutoCorr(ftimes, update= False, **kwargs)
        corrMap= graphData['corrMap']
        corrMap[np.isnan(corrMap)]= 0
        xshift= graphData['xshift']
        yshift= graphData['yshift']
        
        pixel= np.int(np.diff(xshift).mean())
        
        ny, nx= corrMap.shape
        rpeaks= np.zeros(corrMap.shape, dtype= bool)
        cpeaks= np.zeros(corrMap.shape, dtype= bool)
        for j in np.arange(ny):
            rpeaks[j, extrema(corrMap[j, :])[1]]= True
        for i in np.arange(nx):
            cpeaks[extrema(corrMap[:, i])[1], i]= True
        ymax, xmax= find2d(np.logical_and(rpeaks, cpeaks))
        
        peakDist= np.sqrt((ymax- find(yshift==0))**2+ (xmax- find(xshift==0))**2)
        sortInd= np.argsort(peakDist)
        ymax, xmax, peakDist= ymax[sortInd], xmax[sortInd], peakDist[sortInd]
        
        ymax, xmax, peakDist= (ymax[1:7], xmax[1:7], peakDist[1:7]) if ymax.size>=7 else ([], [], [])
        theta= np.arctan2(yshift[ymax], xshift[xmax])*180/np.pi
        theta[theta< 0]+= 360
        sortInd= np.argsort(theta)
        ymax, xmax, peakDist, theta= (ymax[sortInd], xmax[sortInd], peakDist[sortInd], theta[sortInd])
        
        graphData['ymax']= yshift[ymax]
        graphData['xmax']= xshift[xmax]
        
        meanDist=  peakDist.mean()
        X, Y= np.meshgrid(xshift, yshift)
        distMat= np.sqrt(X**2+ Y**2)/pixel
        
        if len(ymax)== np.logical_and(peakDist> 0.75*meanDist, peakDist< 1.25*meanDist).sum(): # if all of them are within tolerance(25%)
            maskInd= np.logical_and(distMat> 0.5*meanDist, distMat< 1.5*meanDist)
            rotCorr= np.array([ corrCoeff(rot2d(corrMap, theta)[maskInd], corrMap[maskInd]) for k, theta in enumerate(bins)])
            ramax, rimax, ramin, rimin= extrema(rotCorr)             
            mThetaPk, mThetaTr= (np.diff(bins[rimax]).mean(), np.diff(bins[rimin]).mean()) if rimax.size and rimin.size else (None, None)
            graphData['rimax']= rimax      
            graphData['rimin']= rimin
            graphData['anglemax']= bins[rimax]
            graphData['anglemin']= bins[rimin]
            graphData['rotAngle']= bins
            graphData['rotCorr']= rotCorr
        
            if mThetaPk is not None and mThetaTr is not None:            
                isGrid= True if 60- tol < mThetaPk < 60+ tol and 60- tol < mThetaTr < 60+ tol else False
            else:
                isGrid= False
            
            meanAlpha= np.diff(theta).mean()            
            psi= theta[np.array([2, 3, 4, 5, 0, 1])]- theta
            psi[psi<0]+= 360
            meanPsi= psi.mean()
            
            _results['isGrid']= isGrid and 120- tol< meanPsi< 120+ tol and 60- tol< meanAlpha< 60+ tol
            _results['meanAlpha']= meanAlpha
            _results['meanPsi']= meanPsi
            _results['gridSpacing']= meanDist*pixel
            _results['gridScore']= rotCorr[rimax].max()- rotCorr[rimin].min() # Difference between highest Pearson R at peaks and lowest at troughs
            _results['gridOrientation']= theta[0]
                                                     
        else:
            _results['isGrid']= False
            
        self.updateResult(_results)
        return graphData
        
    def MRA(self, ftimes, **kwargs):
        _results= oDict()
        graphData= oDict()
        subsampInterv= kwargs.get('subsampInterv', 0.1)
        episode= kwargs.get('episode', 120)
        nrep= kwargs.get('nrep', 1000)
        sampRate= 1/subsampInterv
        stamp= 1/sampRate
        time= np.arange(0, self.getDuration(),  stamp)
        Y= histogram(ftimes, time)[0]* sampRate # Instant firing rate
        
        nt= time.size
        xloc, yloc, loc, hd, speed, angVel, distBorder= list(np.zeros((7, nt)))
        tmp= self.place(ftimes)
        placeRate, xedges, yedges= (tmp['smoothMap'], tmp['xedges'], tmp['yedges'])
        placeRate[np.isnan(placeRate)]= 0
        for i in np.arange(nt):
            ind= find(np.logical_and(self.getTime()>= time[i], self.getTime()< time[i]+ stamp))
            xloc[i]= np.median(self.getX()[ind])
            yloc[i]= np.median(self.getY()[ind])
            if histogram(yloc[i], yedges)[1]< yedges.size and histogram(xloc[i], xedges)[1]< xedges.size:
                loc[i]= placeRate[histogram(yloc[i], yedges)[1], histogram(xloc[i], xedges)[1]]
            hd[i]= np.median(self.getDirection()[ind])
            speed[i]= np.median(self.getSpeed()[ind])
            angVel[i]= np.median(self.getAngVel()[ind])
            distBorder[i]= np.median(self.getBorder()[0][ind])
            
        tmp= self.hdRate(ftimes, update= False)
        hdRate, hdBins= (tmp['hdRate'], tmp['bins'])
        cs= CircStat()
        cs.setTheta(hd)
        hd= hdRate[cs.circHistogram(hdBins)[1]] # replaced by corresponding rate
        # Speed+ angVel will be linearly modelled, so no transformation required; AngVel will be replaced by the non-linear rate
        tmp= self.border(ftimes, update= False)
        borderRate, borderBins= (tmp['distRate'], tmp['distBins'])
        distBorder= borderRate[histogram(distBorder, borderBins)[1]] # replaced by corresponding rate
        
        ns= int(episode/stamp) # row to select in random
        
        X= np.vstack((loc, hd, speed, angVel, distBorder)).transpose()
        lm= LinearRegression(fit_intercept= True, normalize= True)
        
        Rsq= np.zeros((nrep, 6))
        for i in np.arange(nrep):
            ind= np.random.permutation(time.size)[:ns]
            lm.fit(X[ind, :], Y[ind])
            Rsq[i, 0]= lm.score(X[ind, :], Y[ind])
            for j in np.arange(5):
                varind= np.array([k for k in range(5) if k!=j])
                lm.fit(X[np.ix_(ind, varind)], Y[ind]) #np.ix_ is used for braodcasting the index arrays
                Rsq[i, j+1]= Rsq[i, 0]- lm.score(X[np.ix_(ind, varind)], Y[ind])
        
        meanRsq = Rsq.mean(axis= 0)
        
        varOrder= ['Total', 'Loc', 'HD', 'Speed', 'AngVel', 'DistBorder']
        
        graphData['order']= varOrder
        graphData['Rsq']= Rsq
        graphData['meanRsq']= meanRsq
        graphData['maxRsq']= Rsq.max(axis= 0)
        graphData['minRsq']= Rsq.min(axis= 0)  
        graphData['stdRsq']= Rsq.std(axis= 0)
        _results['multRsq']= meanRsq[0]
        for i, key in enumerate(varOrder):
            if i>0:
                _results['semiRsq'+key]= meanRsq[i]
        self.updateResult(_results)
            
        return graphData
    def interdependence(self, ftimes, **kwargs):
        _results= oDict()
        pixel= kwargs.get('pixel', 3)
        hdbinsize= kwargs.get('hdbinsize', 5)
        spbinsize= kwargs.get('spbinsize', 1)
        sprange= kwargs.get('sprange', [0, 40])
        abinsize= kwargs.get('abinsize', 10)
        angvelrange= kwargs.get('angvelrange', [-500, 500])
        
        placeData= self.place(ftimes, pixel= pixel, update= False)
        fmap= placeData['smoothMap']
        fmap[np.isnan(fmap)]= 0
        xloc= self.getX()
        yloc= self.getY()
        xedges= placeData['xedges']
        yedges= placeData['yedges']
        
        hdData= self.hdRate(ftimes, binsize= hdbinsize, update= False)
        bins= hdData['bins']
        predRate= np.zeros(bins.size)
        for i, b in enumerate(bins):
            ind= np.logical_and(hdData['hd']>= b, hdData['hd']< b+ hdbinsize)
            tmap= histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap/= self.getSamplingRate()
            predRate[i]= np.sum(fmap*tmap)/ tmap.sum()
        _results['DRHP']= np.abs(np.log((1+ hdData['smoothRate'])/ (1+ predRate))).sum()/bins.size
        
        spData= self.speed(ftimes, binsize= spbinsize, range= sprange, update= False)
        bins= spData['bins']
        predRate= np.zeros(bins.size)
        speed= self.getSpeed()
        for i, b in enumerate(bins):
            ind= np.logical_and(speed>= b, speed< b+ spbinsize)
            tmap= histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap/= self.getSamplingRate()
            predRate[i]= np.sum(fmap*tmap)/ tmap.sum()
        _results['DRSP']= np.abs(np.log((1+ spData['rate'])/ (1+ predRate))).sum()/bins.size
                
        angVelData= self.angularVelocity(ftimes, binsize= abinsize, range= angvelrange, update= False)
        bins= np.hstack((angVelData['leftBins'], angVelData['rightBins']))
        predRate= np.zeros(bins.size)
        angVel= self.getAngVel()
        for i, b in enumerate(bins):
            ind= np.logical_and(angVel>= b, angVel< b+ abinsize)
            tmap= histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap/= self.getSamplingRate()
            predRate[i]= np.sum(fmap*tmap)/ tmap.sum()
        angVelObs= np.hstack((angVelData['leftRate'], angVelData['rightRate']))
        _results['DRAP']= np.abs(np.log((1+ angVelObs)/ (1+ predRate))).sum()/bins.size
        
        borderData= self.border(ftimes, update= False)
        bins= borderData['distBins']
        dbinsize= np.diff(bins).mean()
        predRate= np.zeros(bins.size)
        border= self.getBorder()[0]
        for i, b in enumerate(bins):
            ind= np.logical_and(border>= b, border< b+ dbinsize)
            tmap= histogram2d(yloc[ind], xloc[ind], yedges, xedges)[0]
            tmap/= self.getSamplingRate()
            predRate[i]= np.sum(fmap*tmap)/ tmap.sum()
        _results['DRBP']= np.abs(np.log((1+ borderData['distRate'])/ (1+ predRate))).sum()/bins.size
        
        self.updateResult(_results)
        
#    def __getattr__(self, arg):
#        if hasattr(self.spike, arg):
#            return getattr(self.spike, arg)
#        elif hasattr(self.lfp, arg):
#            return getattr(self.lfp, arg)
    
class NLfp(NBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fileTag= ''
        self._channelID= 0
        self._samples= None
        self._timestamp= None
        self._type= 'lfp'
        self.setRecordInfo({'Total samples': 0})
    
    # For multi-unit analysis, {'SpikeName': cell_no} pairs should be used as function input
    def setChannelID(self, channel_ID= ''):
        self._channelID= channel_ID
    def getChannelID (self):
        return self._channelID
    
    def setFileTag(self, fileTag):
        self._fileTag= fileTag
    def getFileTag(self):
        return self._fileTag

    def getTimestamp(self):
        return self._timestamp
    def _setTimestamp(self, timestamp= None):
        if timestamp is not None:
            self._timestamp= timestamp

    def getSamples(self):
        return self._samples
    def _setSamples(self, samples= []):
        self._samples= samples
                        
    def load(self, filename= None, system= None):
        if system is None:
            system= self._system
        else:
            self._system= system
        if filename is None:
            filename= self._filename
        else:
            self._filename= filename
        loader= getattr(self, 'loadLFP_'+ system)
        loader(filename)
        
    def addSpike(self, spike= None, **kwargs):
        new_spike= self._addNode(NSpike, spike, 'spike', **kwargs)
        return new_spike
                
    def loadSpike(self, names= 'all'):
        if names== 'all':
            for spike in self._spikes:
                spike.load()
        else:
            for name in names:
                spike= self.getSpikesByName(name)
                spike.load()
            
    def addLfp(self, lfp= None, **kwargs):
        new_lfp= self._addNode(self.__class__, lfp, 'lfp', **kwargs)
        return new_lfp
        
    def loadLfp(self, names= None):
        if names is None:
            self.load()
        elif names== 'all':
            for lfp in self._lfp:
                lfp.load()
        else:
            for name in names:
                lfp= self.getLfpByName(name)
                lfp.load()               
        
    def spectrum(self, **kwargs):
        graphData= oDict()
        
        Fs= self.getSamplingRate()
        lfp= self.getSamples()
        
        window= kwargs.get('window', 1)
        window= sg.get_window('hann', int(window*Fs)) if isinstance(window, float) else window
        
        win_sec= np.ceil(window.size/Fs)
        
        noverlap= kwargs.get('noverlap', 0.5*win_sec)
        noverlap= noverlap if noverlap< win_sec else 0.5*win_sec
        noverlap= np.ceil(noverlap*Fs)

        nfft= kwargs.get('nfft', 2*Fs)
        nfft= np.power(2, int(np.ceil(np.log2(nfft))))
        
        ptype= kwargs.get('ptype', 'psd')
        ptype= 'spectrum' if ptype== 'power' else 'density'

        prefilt= kwargs.get('prefilt', True)
        _filter= kwargs.get('filtset', [10, 1.5, 40, 'bandpass'])
        
        fmax= kwargs.get('fmax', Fs/2)
        
        if prefilt:
            lfp= butter_filter(lfp, Fs, *_filter)
            
        tr= kwargs.get('tr', False)
        db= kwargs.get('db', False)
        if tr:
            f, t, Sxx= sg.spectrogram(lfp, fs= Fs, \
                    window= window, nperseg= window.size, noverlap= noverlap, nfft= nfft, \
                    detrend='constant', return_onesided=True, scaling= ptype)
            
            graphData['t']= t
            graphData['f']= f[find(f<= fmax)]

            if db:
                Sxx= 10*np.log10(Sxx/np.amax(Sxx))
                Sxx= Sxx.flatten()
                Sxx[find(Sxx < -40)]= -40
                Sxx= np.reshape(Sxx, [f.size, t.size])
            
#            graphData['Sxx']= np.empty([find(f<= fmax).size, t.size])
#            graphData['Sxx']= np.array([Sxx[i, :] for i in find(f<= fmax)])
            graphData['Sxx']= Sxx[find(f<= fmax), :]
        else:
            f, Pxx= sg.welch(lfp, fs= Fs, \
                    window= window, nperseg= window.size, noverlap= noverlap, nfft= nfft, \
                    detrend='constant', return_onesided=True, scaling= ptype)

            graphData['f']= f[find(f<= fmax)]

            if db:
                Pxx= 10*np.log10(Pxx/Pxx.max())
                Pxx[find(Pxx<-40)]= -40
            graphData['Pxx']= Pxx[find(f<= fmax)]
        
        return graphData
        
    def phaseDist(self, eventStamp, **kwargs):
        graphData= oDict()
        
        cs=CircStat()
        
        lfp= self.getSamples()*1000
        Fs= self.getSamplingRate()
        time= self.getTimestamp()
        
        # Input parameters
        bins= int(360/kwargs.get('binsize', 5))
        rbinsize= kwargs.get('rbinsize', 2) # raster binsize
        rbins= int(360/rbinsize)
        fwin= kwargs.get('fwin', [6, 12])
        pratio= kwargs.get('pratio', 0.2) 
        aratio= kwargs.get('aratio', 0.15)

    # Filter                
        fmax= fwin[1]
        fmin= fwin[0]
        _filter= [5, fmin, fmax, 'bandpass']
        _prefilt= kwargs.get('filtset', [10, 1.5, 40, 'bandpass'])
        
        b_lfp= butter_filter(lfp, Fs, *_filter) # band LFP
        lfp= butter_filter(lfp, Fs, *_prefilt)

    # Measure phase        
        hilb= sg.hilbert(b_lfp)
#        self.hilb= hilb
#        phase= np.remainder(np.angle(hilb, deg= True)+ 360, 360)
        phase= np.angle(hilb, deg= True)
        phase[phase< 0]= phase[phase<0]+ 360
        mag= np.abs(hilb)
        
        ephase= np.interp(eventStamp, time, phase)
        
        p2p= np.abs(np.max(lfp)- np.min(lfp))
        xline= 0.5* np.mean(mag) # cross line
        
        # Detection algo
        # zero cross
        mag1= mag[0:-3]
        mag2= mag[1:-2]
        mag3= mag[2:-1]
        
        xind= np.union1d(find(np.logical_and(mag1< xline, mag2> xline)), \
                find(np.logical_and(np.logical_and(mag1< xline, mag2== xline), mag3> xline)))            
        
        # Ignore segments <1/fmax
        i= 0
        rcount= np.empty([0,])
        bcount= np.empty([0, 0])
        
        phBins= np.arange(0, 360, 360/bins)
        rbins= np.arange(0, 360, 360/rbins)
        
        seg_count= 0
        while i< len(xind)-1:
            k= i+1
            while time[xind[k]]- time[xind[i]] < 1/fmin and k< len(xind)-1:
                k+=1
#            print(time[xind[i]], time[xind[k]])
            s_lfp= lfp[xind[i]: xind[k]]        
            s_p2p= np.abs(np.max(s_lfp)- np.min(s_lfp))
            
            if s_p2p>= aratio*p2p:     
                s_psd, f= fft_psd(s_lfp, Fs)
                if np.sum(s_psd[np.logical_and(f>= fmin, f<= fmax)])> pratio* np.sum(s_psd):
                    # Phase distribution
                    s_phase= ephase[np.logical_and(eventStamp> time[xind[i]], eventStamp<= time[xind[k]])]
#                    print(s_phase.shape, s_phase.shape)
                    
                    if not s_phase.shape[0]:
                        pass
                    else:
                        seg_count+=1
                        cs.setTheta(s_phase)
                        temp_count= cs.circHistogram(bins= rbinsize)
#                        temp_count= np.histogram(s_phase, bins= rbins, range= [0, 360])
                        temp_count= temp_count[0]
                        if not rcount.size:
                            rcount= temp_count
                        else:                        
                            rcount=  np.append(rcount, temp_count)
    
                        temp_count= np.histogram(s_phase, bins= bins, range= [0, 360])
                        temp_count= np.resize(temp_count[0], [1, bins])
                        if not len(bcount):
                            bcount= temp_count
                        else:
                            bcount=  np.append(bcount, temp_count, axis= 0)
            i= k
            
        rcount=rcount.reshape([seg_count, rbins.size])
        
        phCount= np.sum(bcount, axis= 0)
        
        cs.setRho(phCount)
        cs.setTheta(phBins)
        
        cs.calcStat()
        result= cs.getResult()
        meanTheta= result['meanTheta']*np.pi/180
        
        graphData['meanTheta']= meanTheta
        graphData['phCount']= phCount
        graphData['phBins']= phBins
        graphData['raster']= rcount
        graphData['rasterbins']= rbins
        
        return graphData
    
    def PLV(self, eventStamp, **kwargs):
        graphData= oDict()
        
        lfp= self.getSamples()*1000
        Fs= self.getSamplingRate()
        time= self.getTimestamp()
        
        window= np.array(kwargs.get('window', [-0.5, 0.5]))
        win= np.ceil(window*Fs).astype(int)
        win= np.arange(win[0], win[1])
        slep_win= sg.hann(win.size, False)
        
        nfft= kwargs.get('nfft', 1024)
        mode= kwargs.get('mode', None) # None, 'bs', 'tr' bs= bootstrp, tr= time-resolved
        fwin= kwargs.get('fwin', [])     
        
        xf= np.arange(0, Fs, Fs/nfft)
        f= xf[0: int(nfft/2)+ 1]
        
        ind= np.arange(f.size) if len(fwin)== 0 else find(np.logical_and(f>= fwin[0], f<= fwin[1]))
        
        if mode== 'bs':
            nsample= kwargs.get('nsample', 50)
            nrep= kwargs.get('nrep', 500)
            
            STA= np.empty([nrep, win.size])
            fSTA= np.empty([nrep, ind.size])
            STP= np.empty([nrep, ind.size])
            SFC= np.empty([nrep, ind.size])
            PLV= np.empty([nrep, ind.size])
            
            for i in np.arange(nrep):
                data= self.PLV(np.random.choice(eventStamp, nsample, False), \
                        window= window, nfft= nfft, mode= None, fwin= fwin)
                t= data['t']
                STA[i, :]= data['STA']
                fSTA[i, :]= data['fSTA']
                STP[i, :]= data['STP']
                SFC[i, :]= data['SFC']
                PLV[i, :]= data['PLV']

            graphData['t']= t
            graphData['f']= f[ind]
            graphData['STAm']= STA.mean(0)
            graphData['fSTAm']= fSTA.mean(0)
            graphData['STPm']= STP.mean(0)
            graphData['SFCm']= SFC.mean(0)
            graphData['PLVm']= PLV.mean(0)
            
            graphData['STAe']= stats.sem(STA, 0)
            graphData['fSTAe']= stats.sem(fSTA, 0)
            graphData['STPe']= stats.sem(STP, 0)
            graphData['SFCe']= stats.sem(SFC, 0)
            graphData['PLVe']= stats.sem(PLV, 0)
                                        
        elif mode== 'tr':
            nsample= kwargs.get('nsample', None)
            
            slide= kwargs.get('slide', 25) # in ms
            slide= slide/1000 # convert to sec
            
            offset= np.arange(window[0], window[-1], slide)
            nwin= offset.size
            
#            STA= np.empty([nwin, win.size])
            fSTA= np.empty([nwin, ind.size])
            STP= np.empty([nwin, ind.size])
            SFC= np.empty([nwin, ind.size])
            PLV= np.empty([nwin, ind.size])
            
            if nsample is None or nsample> eventStamp.size:
                stamp= eventStamp
            else:
                stamp= np.random.choice(eventStamp, nsample, False)            
            
            for i in np.arange(nwin):
                data= self.PLV(stamp + offset[i], \
                        nfft= nfft, mode= None, fwin= fwin, window= window)
                t= data['t']
                fSTA[i, :]= data['fSTA']
                STP[i, :]= data['STP']
                SFC[i, :]= data['SFC']
                PLV[i, :]= data['PLV']

            graphData['offset']= offset
            graphData['f']= f[ind]
            graphData['fSTA']= fSTA.transpose()
            graphData['STP']= STP.transpose()
            graphData['SFC']= SFC.transpose()
            graphData['PLV']= PLV.transpose()
            
        elif mode is None:
            center= time.searchsorted(eventStamp)
            # Keep windows within data
            center= np.array([center[i] for i in range(0, len(eventStamp)) \
                if center[i]+ win[0] >= 0 and center[i]+ win[-1] <= time.size])
            
            sta_data= self.eventTA(eventStamp, **kwargs)
            STA= sta_data['ETA']
            
            fSTA= fft(np.multiply(STA, slep_win), nfft)

            fSTA= np.absolute(fSTA[0: int(nfft/2)+ 1])**2/nfft**2
            fSTA[1:-1]= 2*fSTA[1:-1]            
            
            fLFP= np.array([fft(np.multiply(lfp[x+ win], slep_win), nfft) \
                    for x in center])
            
            STP= np.absolute(fLFP[:, 0: int(nfft/2)+ 1])**2/nfft**2            
            STP[:, 1:-1]= 2*STP[:, 1:-1]
            STP= STP.mean(0)
            
            SFC= np.divide(fSTA, STP)*100
            
            PLV= np.copy(fLFP)
            
            # Normalize
            PLV= np.divide(PLV, np.absolute(PLV))
            PLV[np.isnan(PLV)]= 0
            
            PLV= np.absolute(PLV.mean(0))[0: int(nfft/2)+ 1]
            PLV[1:-1]= 2*PLV[1:-1]
            
            graphData['t']= sta_data['t']
            graphData['f']= f[ind]
            graphData['STA']= STA
            graphData['fSTA']= fSTA[ind]
            graphData['STP']= STP[ind]
            graphData['SFC']= SFC[ind]
            graphData['PLV']= PLV[ind]

        return graphData
    
    def eventTA(self, eventStamp =None, **kwargs):
        graphData= oDict()
        window= np.array(kwargs.get('window', [-0.5, 0.5]))
#        mode= kwargs.get('mode', None)

        if eventStamp is None:
            spike= kwargs.get('spike', None)
            if isinstance(spike, NSpike):
               eventStamp= spike.getUnitStamp() 
            elif spike in self.getSpikeNames():
                eventStamp= self.getSpike(spike).getUnitStamp()
        if eventStamp is None:
            logging.error('No valid event timestamp or spike is provided')
        else:
            lfp= self.getSamples()*1000
            Fs= self.getSamplingRate()
            time= self.getTimestamp()
            center= time.searchsorted(eventStamp, side= 'left')
            win= np.ceil(window*Fs).astype(int)
            win= np.arange(win[0], win[1])
            
            # Keep windows within data
            center= np.array([center[i] for i in range(0, len(eventStamp)) \
                if center[i]+ win[0] >= 0 and center[i]+ win[-1] <= time.size])
            
            eta= reduce(lambda y, x: y+ lfp[x+ win], center)
            eta= eta/center.size
            
            graphData['t']= win/Fs
            graphData['ETA'] = eta
            graphData['center']= center
            
        return graphData
        
    def spikeLfpCausality(self, spike= None, **kwargs):
        # Check if the spike exists in the lfp's own spike list, if not if this is an event stream?
        pass
    
    def save_to_hdf5(self, file_name= None, system= None):
        hdf= Nhdf()
        if file_name and system:
            if os.path.exists(file_name) :
                self.setFilename(file_name)
                self.setSystem(system)
                self.load()
            else:
                logging.error('Specified file cannot be found!')                                
        
        hdf.save_lfp(lfp= self)
        hdf.close()

    def loadLFP_NWB(self, file_name):
        file_name, path= file_name.split('+')
        if os.path.exists(file_name):
            hdf= Nhdf()
            hdf.setFilename(file_name)
            self= hdf.load_lfp(path= path, lfp= self)
            hdf.close()
    
    def loadLFP_Axona (self, file_name):
    
        words= file_name.split(sep= os.sep)
        file_directory= os.sep.join(words[0:-1])
        file_tag, file_extension= words[-1].split('.')
        set_file= file_directory + os.sep + file_tag + '.set'
        
        self._setDataSource(file_name)
        self._setSourceFormat('Axona')
        
        with open(file_name, 'rb') as f:
            while True:
                line= f.readline()
                try:
                    line= line.decode('UTF-8')
                except:
                    break
                    
                if ""== line:
                    break
                if line.startswith('trial_date'):                
                    self._setDate(' '.join(line.replace(',', ' ').split()[1:]))
                if line.startswith('trial_time'):
                    self._setTime(line.split()[1])
                if line.startswith('experimenter'):
                    self._setExperiemnter( ' '.join(line.split()[1:]))
                if line.startswith('comments'):                
                    self._setComments( ' '.join(line.split()[1:]))
                if line.startswith('duration'):
                    self._setDuration( float(''.join(line.split()[1:])))
                if line.startswith('sw_version'):
                    self._setFileVersion( line.split()[1])
                if line.startswith('num_chans'):
                    self._setTotalChannel( int(''.join(line.split()[1:])))
                if line.startswith('sample_rate'):
                    self._setSamplingRate( float(''.join(re.findall(r'\d+.\d+|\d+', line))))
                if line.startswith('bytes_per_sample'):
                    self._setBytesPerSample( int(''.join(line.split()[1:])))
                if line.startswith('num_'+ file_extension.upper() + '_samples'):
                    self._setTotalSamples( int(''.join(line.split()[1:])))
            
            num_samples= self.getTotalSamples()
            bytes_per_sample= self.getBytesPerSample()
            
            f.seek(0, 0)
            header_offset= [];
            while True:
                try:
                    buff= f.read(10).decode('UTF-8')                    
                except:
                    break
                if buff == 'data_start':
                    header_offset= f.tell()
                    break
                else:
                    f.seek(-9, 1)
    
            eeg_ID= re.findall(r'\d+', file_extension)
            self.setFileTag (1 if not eeg_ID else int(eeg_ID[0]))
            max_ADC_count= 2**(8*bytes_per_sample-1)-1
            max_byte_value= 2**(8*bytes_per_sample)
            
            with open(set_file, 'r') as f_set:
                lines= f_set.readlines()
                channel_lines= dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                            for line in lines if line.startswith('EEG_ch_')])
                channel_ID= channel_lines[self.getFileTag()]
                self.setChannelID( channel_ID)
                
                gain_lines= dict([tuple(map(int, re.findall(r'\d+.\d+|\d+', line)[0].split()))\
                        for line in lines if 'gain_ch_' in line])
                gain= gain_lines[channel_ID-1]
                
                for line in lines:
                    if line.startswith('ADC_fullscale_mv'):
                        self._setFullscaleMv( int(re.findall(r'\d+.\d+|d+', line)[0]))
                        break
                AD_bit_uvolt= 2*self.getFullscaleMv()/ \
                                 (gain*np.power(2, 8*bytes_per_sample))            
            
            record_size= bytes_per_sample
            sample_le= 256**(np.arange(0, bytes_per_sample, 1))
                    
            if not header_offset:
                print('Error: data_start marker not found!')
            else:
                f.seek(header_offset, 0)
                byte_buffer= np.fromfile(f, dtype= 'uint8')
                len_bytebuffer= len(byte_buffer)
                end_offset= len('\r\ndata_end\r')
                lfp_wave= np.zeros([num_samples, ], dtype= np.float32)
                for k in np.arange(0, bytes_per_sample, 1):
                    byte_offset= k
                    sample_value= (sample_le[k]* byte_buffer[byte_offset \
                                  :byte_offset+ len_bytebuffer- end_offset- record_size\
                                  :record_size])
                    if sample_value.size< num_samples:
                        sample_value= np.append(sample_value, np.zeros([num_samples-sample_value.size,]))
                    sample_value= sample_value.astype(np.float32, casting= 'unsafe', copy= False)
                    np.add(lfp_wave, sample_value, out= lfp_wave)
                np.putmask(lfp_wave, lfp_wave> max_ADC_count, lfp_wave- max_byte_value)
                
                self._setSamples( lfp_wave*AD_bit_uvolt)
                self._setTimestamp( np.arange(0, num_samples, 1)/self.getSamplingRate())
    
    def loadLFP_Neuralynx (self, file_name):
#        file_name= 'C:\\Users\\Raju\\Google Drive\\Sample Data for NC\\Sample NLX Data Set\\' + 'CSC1.ncs'
        
        self._setDataSource(file_name)
        self._setSourceFormat('Neuralynx')
        
        # Format description for the NLX file:
        
        resamp_freq= 250 # NeuroChaT subsamples the original recording from 32000 to 250
        
        header_offset= 16*1024 # fixed for NLX files
        
        bytes_per_timestamp= 8
        bytes_chan_no= 4 
        bytes_sample_freq= 4
        bytes_num_valid_samples= 4
        bytes_per_sample= 2       
        samples_per_record= 512
        
        max_byte_value= np.power(2, bytes_per_sample*8)
        max_ADC_count= np.power(2, bytes_per_sample*8- 1)-1
        AD_bit_uvolt= 10**-6

        self._setBytesPerSample(bytes_per_sample)
        
        record_size= None
        with open(file_name, 'rb') as f:
            while True:                
                line= f.readline()
                try:
                    line= line.decode('UTF-8')
                except:
                    break
                    
                if ""== line:
                    break
                if 'SamplingFrequency' in line:
                    self._setSamplingRate( float(''.join(re.findall(r'\d+.\d+|\d+', line)))) # We are subsampling from the blocks of 512 samples per record
                if 'RecordSize' in line:
                    record_size= int(''.join(re.findall(r'\d+.\d+|\d+', line)))
                if 'Time Opened' in line:          
                    self._setDate(re.search(r'\d+/\d+/\d+', line).group())
                    self._setTime(re.search(r'\d+:\d+:\d+', line).group())
                if 'FileVersion' in line:
                    self._setFileVersion(line.split()[1])
                if 'ADMaxValue' in line:
                    max_ADC_count= float(''.join(re.findall(r'\d+.\d+|\d+', line)))             
                if 'ADBitVolts' in line:
                    AD_bit_uvolt= float(''.join(re.findall(r'\d+.\d+|\d+', line)))*(10**6)
                    
            self._setFullscaleMv(max_byte_value*AD_bit_uvolt/2) # gain= 1 assumed to keep in similarity to Axona
                
            if not record_size:
                record_size= bytes_per_timestamp+ \
                             bytes_chan_no+ \
                             bytes_sample_freq+ \
                             bytes_num_valid_samples+ \
                             bytes_per_sample*samples_per_record
             
            time_offset= 0
            sample_freq_offset= bytes_per_timestamp+ bytes_chan_no
            num_valid_samples_offset= sample_freq_offset+ bytes_sample_freq
            sample_offset= num_valid_samples_offset+ bytes_num_valid_samples
            f.seek(0, 2)
            num_samples= int((f.tell()- header_offset)/record_size)
            
            f.seek(header_offset, 0)
            time= np.array([])
            lfp_wave= np.array([])
            sample_le= 256**(np.arange(0, bytes_per_sample, 1))
            for i in np.arange(num_samples):
                sample_bytes= np.fromfile(f, dtype= 'uint8', count= record_size)
                block_start= int.from_bytes(sample_bytes[time_offset+ \
                                np.arange(bytes_per_timestamp)], byteorder= 'little', signed= False)/10**6
                valid_samples= int.from_bytes(sample_bytes[num_valid_samples_offset+ \
                                np.arange(bytes_num_valid_samples)], byteorder= 'little', signed= False)
                sampling_freq= int.from_bytes(sample_bytes[sample_freq_offset+ \
                                np.arange(bytes_sample_freq)], byteorder= 'little', signed= False)

                wave_bytes= sample_bytes[sample_offset+ np.arange(valid_samples* bytes_per_sample)]\
                                .reshape([valid_samples, bytes_per_sample])
                block_wave= np.dot(wave_bytes, sample_le)
#                for k in np.arange(valid_samples):
#                    block_wave[k]= int.from_bytes(sample_bytes[sample_offset+ k*bytes_per_sample+ \
#                                np.arange(bytes_per_sample)], byteorder= 'little', signed= False)
                np.putmask(block_wave,block_wave> max_ADC_count, block_wave- max_byte_value)
                block_wave= block_wave*AD_bit_uvolt
                block_time= block_start+  np.arange(valid_samples)/ sampling_freq
                interp_time= np.arange(block_start, block_time[-1], 1/resamp_freq)
                interp_wave= np.interp(interp_time, block_time, block_wave)
                time= np.append(time, interp_time)
                lfp_wave= np.append(lfp_wave, interp_wave)
            time-= time.min()
            self._setSamples( lfp_wave)
            self._setTotalSamples(lfp_wave.size)
            self._setTimestamp( time)
            self._setSamplingRate(resamp_freq)
            
    def _setTotalSamples(self, tot_samples= 0):
        self._recordInfo['No of samples']= tot_samples
    def _setTotalChannel(self, tot_channels):
        self._recordInfo['No of channels']= tot_channels
    def _setTimestampBytes(self, bytes_per_timestamp):
        self._recordInfo['Bytes per timestamp']= bytes_per_timestamp
    def _setSamplingRate(self, sampling_rate):
        self._recordInfo['Sampling rate']= sampling_rate
    def _setBytesPerSample(self, bytes_per_sample):
        self._recordInfo['Bytes per sample']= bytes_per_sample
    def _setFullscaleMv(self, adc_fullscale_mv):
        self._recordInfo['ADC Fullscale mv']= adc_fullscale_mv    
            
    def getTotalSamples(self):
        return self._recordInfo['No of samples']
    def getTotalChannel(self):
        return self._recordInfo['No of channels']
    def getTimestampBytes(self):
        return self._recordInfo['Bytes per timestamp']
    def getSamplingRate(self):
        return self._recordInfo['Sampling rate']
    def getBytesPerSample(self):
        return self._recordInfo['Bytes per sample']
    def getFullscaleMv(self):
        return self._recordInfo['ADC Fullscale mv']
            
class NData():
    def __init__(self):
        super().__init__()
        self.spike= NSpike(name= 'C0')
        self.spatial= NSpatial( name= 'S0')
        self.lfp= NLfp(name= 'L0')
        self.dataFormat= 'Axona'
        self._results= oDict()
    def getResults(self):
        return self._results
    def updateResults(self, _results):
        self._results.update(_results)
    def resetResults(self):
        self._results= oDict()
#        self.spike.resetResults()
#        self.spatial.resetResults()
#        self.lfp.resetResults()
        
    def getDataFormat(self):
        return self.dataFormat
        
    def setDataFormat(self, dataFormat= None):
        if dataFormat is None:
            dataFormat= self.getDataFormat()
        self.dataFormat= dataFormat            
        self.spike.setSystem(dataFormat)
        self.spatial.setSystem(dataFormat)
        self.lfp.setSystem(dataFormat)
        
    def load(self, system= 'Axona'):
        self.loadSpike()
        self.loadSpatial()
        self.loadLfp()
    
    def save_to_hdf5(self):
        hdf= Nhdf()
        try:
            hdf.save_spike(spike=  self.spike)
        except:
            logging.warning('Error in exporting NSpike data from NData object to the hdf5 file!')
            
        try:
            hdf.save_spatial(self.spatial)
        except:
            logging.warning('Error in exporting NSpatial data from NData object to the hdf5 file!')

        try:
            hdf.save_lfp(self.lfp)
        except:
            logging.warning('Error in exporting NLfp data from NData object to the hdf5 file!')
        
        hdf.close()
            
    def setUnitNo(self, unit_no):
        self.spike.setUnitNo(unit_no)        
    def setSpikeName(self, name= 'C0'):
        self.spike.setName(name)        
    def setSpikeFile(self, filename):
        self.spike.setFilename(filename)
    def getSpikeFile(self):
        return self.spike.getFilename()        
    def loadSpike(self):
        self.spike.load()
        
    def setSpatialFile(self, filename):
        self.spatial.setFilename(filename)
    def getSpatialFile(self):
        return self.spatial.getFilename()
    def setSpatialName(self, name):
        self.spatial.setName(name)
    def loadSpatial(self):
        self.spatial.load()
        
    def setLfpFile(self, filename):
        self.lfp.setFilename(filename)
    def getLfpFile(self):
        return self.lfp.getFilename()        
    def setLfpName(self, name):
        self.lfp.setName(name)
    def loadLfp(self):
        self.lfp.load()
    
    # Forwarding to analysis    
    def waveProperty(self):
        gdata= self.spike.waveProperty()
        self.updateResults(self.spike.getResults())
        return gdata
    def isi(self, bins= 'auto', bound= None, density= False):
        gdata= self.spike.isi(bins, bound, density)
        self.updateResults(self.spike.getResults())
        return gdata
    def isiAutoCorr(self, spike= None, **kwargs):
        gdata= self.spike.isiCorr(spike, **kwargs)
        return gdata
    def burst(self, burstThresh= 5, ibiThresh= 50):
        gdata= self.spike.burst(burstThresh, ibiThresh)
        self.updateResults(self.spike.getResults())
        return gdata
    def thetaIndex(self, **kwargs):
        gdata= self.spike.thetaIndex(**kwargs)
        self.updateResults(self.spike.getResults())
        return gdata         
    def thetaSkipIndex(self, **kwargs):
        gdata= self.spike.thetaSkipIndex(**kwargs)         
        self.updateResults(self.spike.getResults())
        return gdata
    def spectrum(self, **kwargs):
        gdata= self.lfp.spectrum(**kwargs)
        return gdata     
    def phaseDist(self, **kwargs):
        gdata= self.lfp.phaseDist(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.lfp.getResults())
        return gdata
    def PLV(self, **kwargs):        
        gdata= self.lfp.PLV(self.spike.getUnitStamp(), **kwargs)
        return gdata        
    def eventTA(self, **kwargs): 
        gdata= self.lfp.eventTA(self.spike.getUnitStamp(), **kwargs)
        return gdata         
    def spikeLfpCausality(self, **kwargs): 
        gdata= self.lfp.spikeLfpCausality(self.spike.getUnitStamp(), **kwargs)            
        self.updateResults(self.lfp.getResults())
        return gdata         
          
    def speed(self, **kwargs):
        gdata= self.spatial.speed(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata
    def angularVelocity(self, **kwargs):
        gdata= self.spatial.angularVelocity(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata
    def place(self, **kwargs):
        gdata= self.spatial.place(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def locTimeLapse(self, **kwargs):
        gdata= self.spatial.locTimeLapse(self.spike.getUnitStamp(), **kwargs)
        return gdata        
    def locShuffle(self, **kwargs):
        gdata= self.spatial.locShuffle(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata
    def locShift(self, shiftInd= np.arange(-10, 11), **kwargs):
        gdata= self.spatial.locShift(self.spike.getUnitStamp(), shiftInd= shiftInd, **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def locAutoCorr(self, **kwargs):
        gdata= self.spatial.locAutoCorr(self.spike.getUnitStamp(), **kwargs)
        return gdata        
    def locRotCorr(self, **kwargs):
        gdata= self.spatial.locRotCorr(self.spike.getUnitStamp(), **kwargs)
        return gdata        
    def hdRate(self, **kwargs):
        gdata= self.spatial.hdRate(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def hdRateCCW(self, **kwargs):
        gdata= self.spatial.hdRateCCW(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def hdTimeLapse(self):
        gdata= self.spatial.hdTimeLapse(self.spike.getUnitStamp())
        return gdata     
    def hdShuffle(self, **kwargs):
        gdata= self.spatial.hdShuffle(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def hdShift(self, shiftInd= np.arange(-10, 11)):
        gdata= self.spatial.hdShift(self.spike.getUnitStamp(), shiftInd= shiftInd)
        self.updateResults(self.spatial.getResults())
        return gdata
    def border(self, **kwargs):
        gdata= self.spatial.border(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def gradient(self, **kwargs):
        gdata= self.spatial.gradient(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def grid(self, **kwargs):
        gdata= self.spatial.grid(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def MRA(self, **kwargs):
        gdata= self.spatial.MRA(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        return gdata        
    def interdependence(self, **kwargs):
        self.spatial.interdependence(self.spike.getUnitStamp(), **kwargs)
        self.updateResults(self.spatial.getResults())
        
    def __getattr__(self, arg):
        if hasattr(self.spike, arg):
            return getattr(self.spike, arg)
        elif hasattr(self.lfp, arg):
            return getattr(self.lfp, arg) 
        elif hasattr(self.spatial, arg):
            return getattr(self.spatial, arg)            
        else:
            logging.warning('No '+ arg+ ' method or attribute in NeuroData or in composing data class')
            
            
            
class NClust(NBase):
    def __init__(self, **kwargs):
        spike= kwargs.get('spike', None)
        self.wavetime= []
        self.UPSAMPLED= False
        self.ALLIGNED= False
        self.NULL_CHAN_REMOVED= False
        
        if isinstance(spike, NSpike):
            self.spike= spike
        else:
            self.spike= NSpike(**kwargs)
        super().__init__(**kwargs)
        
    def getUnitTags(self):
        return self.spike.getUnitTags()
    
    def setUnitTags(self, new_tags):
        self.spike.setUnitTags()
        
    def getUnitList(self):
        return self.spike.getUnitList()
    
    def _setUnitList(self):
        self.spike._setUnitList()
        
    def getTimestamp(self, unitNo= None):
        self.spike.getTimestamp(unitNo= unitNo)
        
    def getUnitSpikesCount(self, unit_no= None):
        return self.spike.getUnitSpikesCount(unit_no= unit_no)
    
    def getWaveform(self):
        return self.spike.getWaveform()
        
    def _setWaveform(self, spike_waves= []):
        self.spike._setWaveform(spike_waves= spike_waves)

    def getUnitWaves(self, unit_no= None):
        return self.spike.getUnitWaves(unit_no= unit_no)
    
    # For multi-unit analysis, {'SpikeName': cell_no} pairs should be used as function input
                
    def load(self, filename= None, system= None):
        self.spike.load(filename= filename, system= system)
        
    def addSpike(self, spike= None, **kwargs):
        return self.spike.addSpike(spike= spike, **kwargs)
         
    def loadSpike(self, names= None):
        self.spike.loadSpike(names= names)
    
    def waveProperty(self):
        return self.spike.waveProperty()
        
    def isi(self, bins= 'auto', bound= None, density= False):
        return self.spike.isi( bins= bins, bound= bound, density= density)
    
    def isiCorr(self, **kwargs):
        return self.spike.isiCorr(**kwargs)
        
    def psth(self, eventStamp, **kwargs):
        return self.spike.psth(eventStamp, **kwargs)
        
    def burst(self, burstThresh= 5, ibiThresh= 50):
        self.spike.burst(burstThresh= burstThresh, ibiThresh= ibiThresh)
            
    def getTotalSpikes(self):
        return self.spike.getTotalSpikes()
    
    def getTotalChannels(self):
        return self.spike.getTotalChannels()
    
    def getChannelIDs(self):
        return self.spike.getChannelIDs()
    
    def getTimebase(self):
       return  self.spike.getTimebase()
   
    def getSamplingRate(self):
        return self.spike.getSamplingRate()
    
    def getSamplesPerSpike(self):
        return self.spike.getSamplesPerSpike()
    
    def getWaveTimestamp(self):
        # resturn as microsecond
        fs= self.spike.getSamplingRate()/10**6 # fs downsampled so that the time is given in microsecond
        return 1/fs
            
    def save_to_hdf5(self, file_name= None, system= None):
        self.spike.save_to_hdf5()
    
    def get_feat(self, feat_list= ['pc', 'max', 'min'], Npc= 2):
        if not self.NULL_CHAN_REMOVED:
            self.remove_null_chan()
        if not self.ALLIGNED:
            self.align_wave_peak()
            
        trough, trough_loc= self.get_min_wave_chan() # trough only in max_channel
        peak, peak_chan, peak_loc= self.get_max_wave_chan()
#        amp= np.abs(peak-trough)
        pc= self.get_wave_pc(Npc= Npc)
        shape= (self.getTotalSpikes(), 1)
        feat= np.concatenate((peak.reshape(shape), trough.reshape(shape), pc), axis= 1)
        
        return feat
        
        # for all units
    def get_feat_by_unit(self, unit_no= None, feat_list=[]):
        if unit_no in self.getUnitList():
            feat= self.get_feat()
            return feat[self.getUnitTags()== unit_no, :]
        # unit_no mandatory input, error otherwise, just take the ones for that unit from get_feat
    def get_wave_peaks(self):
        wave= self.getWaveform()
        peak= np.zeros((self.getTotalSpikes(), len(wave.keys())))
        peak_loc= np.zeros((self.getTotalSpikes(), len(wave.keys())), dtype= int)
        for i, key in enumerate(wave.keys()):
            peak[:, i]=  np.amax(wave[key], axis= 1)
            peak_loc[:, i]=  np.argmax(wave[key], axis= 1)
        
        return peak, peak_loc
    
    def get_max_wave_chan(self):
        # Peak value at the highest channel, the highest channel, and the index of the peak
        peak, peak_loc= self.get_wave_peaks()
        max_wave_chan= np.argmax(peak, axis= 1)
        max_wave_val= np.amax(peak, axis= 1)
        return max_wave_val, max_wave_chan, peak_loc[np.arange(len(peak_loc)), max_wave_chan]
        
    def align_wave_peak(self, reach= 300, factor= 2):
        if not self.UPSAMPLED:
            self.resample_wave(factor= factor)
        if not self.ALLIGNED:
            shift= round(reach/self.getWaveTimestamp()) # maximum 300microsecond allowed for shift
            wave= self.getWaveform() # NC waves are stroed in waves['ch1'], waves['ch2'] etc. ways
            maxInd= shift+ self.get_max_wave_chan()[2]
            shiftInd= int(np.median(maxInd))- maxInd
            shiftInd[np.abs(shiftInd)> shift]= 0
            stacked_chan= np.empty((self.getTotalSpikes(), self.getSamplesPerSpike(), self.getTotalChannels()))
            keys= []
            i= 0
            for key, val in wave.items():
                stacked_chan[:, :, i]= val
                keys.append(key)
                i+=1
                
            stacked_chan= np.lib.pad(stacked_chan, [(0, 0), (shift, shift), (0, 0)], 'edge')
            
            stacked_chan= np.array([np.roll(stacked_chan[i, :, :], shiftInd[i], axis= 0)[shift: shift+ self.getSamplesPerSpike()] for i in np.arange(shiftInd.size)])
            
            for i, key in enumerate(keys):
                wave[key]= stacked_chan[:, :, i]
            self._setWaveform(wave)
            self.ALLIGNED= True
            
#        stacked_chan= np.roll(stacked_chan, tuple(shiftInd), axis= tuple(np.ones(shiftInd.size)))
            
    def get_wave_min(self):
        wave= self.getWaveform()
        min_w= np.zeros((self.getTotalSpikes(), len(wave.keys())))
        min_loc= np.zeros((self.getTotalSpikes(), len(wave.keys())))
        for i, key in enumerate(wave.keys()):
            min_w[:, i]=  np.amin(wave[key], axis= 1)
            min_loc[:, i]=  np.argmin(wave[key], axis= 1)
        
        return min_w, min_loc
    
    def get_min_wave_chan(self):
        # Peak value at the highest channel, the highest channel, and the index of the peak
        max_wave_chan= self.get_max_wave_chan()[1]
        trough, trough_loc= self.get_wave_min()
        return trough[np.arange(len(max_wave_chan)), max_wave_chan], \
                trough_loc[np.arange(len(max_wave_chan)), max_wave_chan]    
        # use get_max_channel to determine 
    
    def get_wave_pc(self, Npc= 2):
        wave= self.getWaveform()
        pc= np.array([])
        for key, w in wave.items():
            pca= PCA(n_components= 5)
            w_new= pca.fit_transform(w)
            pc_var= pca.explained_variance_ratio_
            
            if Npc and Npc< w_new.shape[1]:
                w_new=  w_new[:, :Npc]
            else:
                w_new= w_new[:, 0:(find(np.cumsum(pc_var)>= 0.95, 1, 'first')[0]+1)]
            if not len(pc):
                pc= w_new
            else:
                pc= np.append(pc, w_new, axis= 1)
        return pc
    
    def get_wavetime(self):
        # calculate the wavetime from the sampling rate and number of sample, returns in microsecond
        nsamp= self.spike.getSamplesPerSpike()
        timestamp= self.getWaveTimestamp()
        return np.arange(0, (nsamp)*timestamp, timestamp)
    
    def resample_wavetime(self, factor= 2):
        wavetime= self.get_wavetime()
        timestamp= self.getWaveTimestamp()
        return np.arange(0, wavetime[-1], timestamp/factor) 
        # return resampled time 
        
    def resample_wave(self, factor= 2):
        # resample wave using spline interpolation using the resampled_time,return wave
        if not self.UPSAMPLED:
            wavetime= self.get_wavetime()
            uptime= self.resample_wavetime(factor=  factor)
            wave= self.getWaveform()
            for key, w in wave.items():
                f= sc.interpolate.interp1d(wavetime, w, axis= 1, kind= 'quadratic')
                wave[key]= f(uptime)
            
            self.spike._setSamplingRate(self.getSamplingRate()*factor)
            self.spike._setSamplesPerSpike(uptime.size)
            self.UPSAMPLED= True
            return wave, uptime
        else:
            logging.warning('You can upsample only once. Please reload data from source file for changing sampling factor!')
            
    def get_wave_energy(self):
        wave= self.getWaveform()
        energy= np.zeros((self.getTotalSpikes(), len(wave.keys())))
        for i, key in enumerate(wave.keys()):
            energy[:, i]= np.sum(np.square(wave[key]), 1)/10**6 # taken the enrgy in mV2
        return energy
    
    def get_max_energy_chan(self):
        energy= self.get_wave_energy()
        return np.argmax(energy, axis= 1)
    
    def remove_null_chan(self):
        wave= self.getWaveform()
        off_chan= []
        for key, w in wave.items():
            if np.abs(w).sum()==0:
                off_chan.append(key)
        if off_chan:
            for key in off_chan:
                del wave[key]
            self._setWaveform(wave)
            self.NULL_CHAN_REMOVED= True
        return off_chan
        # simply detect in which channel everything is zero, which means it's a reference channel or nothing is recorded here
        
    def cluster_separation(self, unit_no= 0):
        # if unit_no==0 all units, matrix output for pairwise comparison, 
        # else maximum BC for the specified unit
        feat= self.get_feat()
        unit_list= self.getUnitList()
        n_units= len(unit_list)

        if unit_no== 0:
            bc= np.zeros([n_units, n_units])
            dh= np.zeros([n_units, n_units])
            for c1 in np.arange(n_units):
                for c2 in np.arange(n_units):
                    X1= feat[self.getUnitTags()== unit_list[c1], :]
                    X2= feat[self.getUnitTags()== unit_list[c2], :]
                    bc[c1, c2]= bhatt(X1, X2)[0]
                    dh[c1, c2]= hellinger(X1, X2)
                    unit_list= self.getUnitList()
        else:
            bc= np.zeros(n_units)
            dh= np.zeros(n_units)
            X1= feat[self.getUnitTags()== unit_no, :]
            for c2 in np.arange(n_units):
                if c2== unit_no:
                    bc[c2]= 0
                    dh[c2]= 1
                else:                    
                    X2= feat[self.getUnitTags()== unit_list[c2], :]
                    bc[c2]= bhatt(X1, X2)[0]
                    dh[c2]= hellinger(X1, X2)
                idx= find(np.array(unit_list)!= unit_no)
            
            return bc[idx], dh[idx]
    def cluster_similarity(self, nclust= None, unit_1= None, unit_2=  None):
        if isinstance(nclust, NClust):
            if unit_1 in self.getUnitList() and unit_2 in nclust.getUnitList():
                X1= self.get_feat_by_unit(unit_no= unit_1)
                X2= nclust.get_feat_by_unit(unit_no= unit_2)
                bc= bhatt(X1, X2)[0]
                dh= hellinger(X1, X2)
        return bc, dh
        
#    @staticmethod
#    def calc_Lratio(X1, X2):
#        pass
#    
#    @staticmethod
#    def calc_ChiSim(X1, X2):
#        pass
        
        # return chi_sim and Enclosement ratio