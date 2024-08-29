import tensorflow as tf
import numpy as np
_strategy = tf.distribute.MirroredStrategy()
def CSLYOLO(input_shape,labels_len,fpn_filters=25,fpn_repeat=1,freeze=True):
    global _strategy
    with _strategy.scope():
        input_ts=tf.keras.Input(shape=input_shape)
        bacbone_outputs=CSLBoneBody(input_ts,freeze=freeze)
        body_outputs=CSLYOLOBody(fpn_filters,fpn_repeat)(*bacbone_outputs)

        l1_anchors=[[0.24737348,0.42670591],[0.12261792,0.23208084],[0.24703822,0.50821878]]
        l2_anchors=[[0.34536429,0.60822857],[0.4659,0.8393],[0.10241657,0.1746698]]
        l3_anchors=[[0.14513302,0.27657725],[0.32344026,0.64453602],[0.41768571,0.74687143]]
        l4_anchors=[[0.1284,0.1923],[0.29715,0.5158],[0.23734027,0.45092207]]
        l5_anchors=[[0.18820261,0.36866516],[0.27787356,0.54488727],[0.4659,0.8393]]
        out_hw_list=list(map(lambda x:[int(x[0]),int(x[1])],[[26,36],[16,22],[7,9],[6,7],[4,5]]))
        anchors_list=[l1_anchors*np.array(out_hw_list[0]),
                  l2_anchors*np.array(out_hw_list[1]),
                  l3_anchors*np.array(out_hw_list[2]),
                  l4_anchors*np.array(out_hw_list[3]),
                  l5_anchors*np.array(out_hw_list[4])]

        net_outputs=CSLConv(anchors_list[0:],labels_len,name="cslconv")(body_outputs[0:])

        model=tf.keras.Model(input_ts,net_outputs)
    return model

def CSLBoneBody(input_ts,freeze=False):
    bacbone_outputs=CSLBone()(input_ts)
    model=tf.keras.Model(input_ts,bacbone_outputs[-1])
    # model.load_weights("weights/cslb_whts.hdf5")
    if(freeze==True):FreezeLayers(model,freeze_type="ALL")
    return bacbone_outputs

def FreezeLayers(model,keys=None,freeze_type="LAYERS"):
    if(freeze_type=="ALL"):
        for layer in model.layers:
            layer.trainable=False
    elif(keys==None):
        raise Exception("FreezeLayers Error: The arg 'keys' can't be None.")
    elif(type(keys[0])==str and freeze_type=="LAYERS"):
        for name in keys:
            model.get_layer(name=name).trainable=False
    elif(type(keys[0])==int and freeze_type=="LAYERS"):
        for idx in keys:
            model.get_layer(index=idx).trainable=False
    return 

class CSLConv(tf.Module):
    def __init__(self,anchors_list,labels_len,name="cslconv"):
        super(CSLConv,self).__init__(name=name)
        self._anchors_list=anchors_list
        self._anchors_num=len(self._anchors_list[0])
        self._labels_len=labels_len
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._outconv=ConvBN(self._anchors_num*(5+self._labels_len+1),
                             kernel_size=(1,1),
                             use_bn=False,
                             activation=None,
                             name=self._name+"_outconv")
    @tf.Module.with_name_scope
    def _Grids(self,featmap_hw):
        featuremap_hight_idx=tf.range(start=0,limit=featmap_hw[0])
        featuremap_hight_idx=tf.expand_dims(featuremap_hight_idx,axis=0)
        featuremap_hight_idx=tf.tile(featuremap_hight_idx,[featmap_hw[1],1])
        featuremap_hight_idx=tf.transpose(featuremap_hight_idx)

        featuremap_width_idx=tf.range(start=0,limit=featmap_hw[1])
        featuremap_width_idx=tf.expand_dims(featuremap_width_idx,axis=0)
        featuremap_width_idx=tf.tile(featuremap_width_idx,[featmap_hw[0],1])

        grids=tf.stack([featuremap_width_idx,featuremap_hight_idx],axis=-1)
        grids=tf.reshape(grids,[1,featmap_hw[0],featmap_hw[1],1,2])
        grids=tf.cast(grids,tf.float32)
        return grids
    @tf.Module.with_name_scope
    def _RestructInTensor(self,input_ts,anchors):
        ftmp_hw=input_ts.get_shape().as_list()[1:3]
        ftmp_wh=tf.cast(tf.reverse(ftmp_hw,[-1]),tf.float32)

        feature_map=self._outconv(input_ts)
        feature_map=tf.reshape(feature_map,[-1,ftmp_hw[0],ftmp_hw[1],self._anchors_num,5+self._labels_len+1])

        box_for_fit=tf.concat([tf.sigmoid(feature_map[...,0:2]),feature_map[...,2:4]],axis=-1)
        pred_xy=(tf.sigmoid(feature_map[...,0:2])+self._Grids(ftmp_hw))/ftmp_wh

        pred_wh=(anchors+feature_map[...,2:4])/ftmp_wh
        pred_wh=pred_wh*tf.cast(pred_wh>=0,tf.float32)
        ones_mask=tf.cast(pred_wh<=1,tf.float32)
        pred_wh=pred_wh*ones_mask+(1-ones_mask)
        
        pred_box=tf.concat([pred_xy,pred_wh],axis=-1)
        pred_cnfd=tf.sigmoid(feature_map[...,4:5])
        pred_classes=tf.sigmoid(feature_map[...,5:])
        output_ts=tf.concat([box_for_fit,pred_box,pred_cnfd,pred_classes],axis=-1)
        return output_ts
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        output_ts_list=[]
        for i,anchors in enumerate(self._anchors_list):
            x=self._RestructInTensor(input_ts_list[i],anchors)
            output_ts_list.append(tf.identity(x))
        return output_ts_list

class CSLMBlock(tf.Module):
    def __init__(self,filters,t,down_rate=1.0,blck_len=1,use_se=True,name="cslmblck"):
        super(CSLMBlock,self).__init__(name=name)
        self._filters=filters
        self._t=t
        self._down_rate=down_rate
        self._blck_len=blck_len
        self._use_se=use_se
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._cslm_list=[]
        self._first_cslm=CSLModule(self._filters,t=self._t,down_rate=self._down_rate,use_se=self._use_se,name=self._name+"_first_cslm")
        for i in range(self._blck_len-1):
            self._cslm_list.append(CSLModule(self._filters,t=self._t,name=self._name+"_cslm_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        first_x=self._first_cslm(input_ts)
        x=first_x
        for i in range(self._blck_len-1):
            x=self._cslm_list[i](x)
        if(self._blck_len>1):
            output_ts=first_x+x
        else:
            output_ts=first_x
        return output_ts


class CSLBone(tf.Module):
    def __init__(self,name="cslbone"):
        super(CSLBone,self).__init__(name=name)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._convbn=ConvBN(4,(3,3),(2,2),name=self._name+"_convbn")
        self._cslmblck_1=CSLMBlock(filters=4,t=1,down_rate=1,blck_len=1,name=self._name+"_cslmblck_1")
        self._cslmblck_2=CSLMBlock(filters=8,t=1,down_rate=0.5,blck_len=1,name=self._name+"_cslmblck_2")
        self._cslmblck_3=CSLMBlock(filters=12,t=1,down_rate=0.5,blck_len=1,name=self._name+"_cslmblck_3")
        self._cslmblck_4=CSLMBlock(filters=18,t=1,down_rate=0.5,blck_len=1,name=self._name+"_cslmblck_4")
        self._cslmblck_5=CSLMBlock(filters=24,t=1,down_rate=0.5,blck_len=1,name=self._name+"_cslmblck_5")
        self._cslmblck_6=CSLMBlock(filters=36,t=1,down_rate=0.5,blck_len=2,name=self._name+"_cslmblck_6")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._convbn(input_ts)
        x=self._cslmblck_1(x)
        x=self._cslmblck_2(x)
        x1=self._cslmblck_3(x)
        x=self._cslmblck_4(x1)
        x2=self._cslmblck_5(x)
        x3=self._cslmblck_6(x2)
        return x1,x2,x3
    
class ConvBN(tf.Module):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="convbn"):
        super(ConvBN,self).__init__()
        self._filters=filters
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._conv=tf.keras.layers.Conv2D(filters=self._filters,
                                          kernel_size=self._kernel_size,
                                          strides=self._strides,
                                          padding=self._padding,
                                          use_bias=self._bias,
                                          name=self._name+"_conv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._conv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class CSLYOLOBody(tf.Module):
    def __init__(self,fpn_filters=96,repeat=3,name="cslyolobody"):
        super(CSLYOLOBody,self).__init__(name=name)
        self._fpn_filters=round(fpn_filters)
        self._repeat=round(repeat)
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._l1_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l1_cspg")
        self._l3_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l3_cspg")
        self._l5_cspg=CSLModule(filters=self._fpn_filters,down_rate=1.0,use_se=True,name=self._name+"_l5_cspg")
        self._l2_bifusion=InputBIFusion(name=self._name+"_l2_bifusion")
        self._l4_bifusion=InputBIFusion(name=self._name+"_l4_bifusion")

        self._cslfpn=CSLFPN(repeat=self._repeat,name=self._name+"_cslfpn")
        # self._vanillafpn=VanillaFPN(name=self._name+"_vanillafpn")
    @tf.Module.with_name_scope
    def __call__(self,bacbone_l1,bacbone_l2,bacbone_l3):
        orig_l1,orig_l2,orig_l3=bacbone_l1,bacbone_l2,bacbone_l3
        l1=self._l1_cspg(orig_l1)
        l3=self._l3_cspg(orig_l2)
        l5=self._l5_cspg(orig_l3)
        l2=self._l2_bifusion(l1,l3)
        l4=self._l4_bifusion(l3,l5)
        l1,l2,l3,l4,l5=self._cslfpn([l1,l2,l3,l4,l5])
        # l1,l2,l3,l4,l5=self._vanillafpn([l1,l2,l3,l4,l5])
        return l1,l2,l3,l4,l5
mish=tf.keras.layers.Lambda(lambda x:x*tf.math.tanh(tf.math.softplus(x)))
hard_sigmoid=tf.keras.layers.Lambda(lambda x:tf.nn.relu6(x+3.0)/6.0)
class CSLModule(tf.Module):
    def __init__(self,filters,t=2,down_rate=1,use_se=False,activation=mish,name="cslmodule"):
        super(CSLModule,self).__init__(name=name)
        self._filters=filters
        self._t=round(t)
        self._down_rate=down_rate
        self._use_se=use_se
        self._activation=activation
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ts):
        input_shape=list(input_ts.shape)
        out_hw=np.array([np.ceil(input_shape[1]*self._down_rate),np.ceil(input_shape[2]*self._down_rate)])

        out_ch_1=self._filters//2
        out_ch_2=self._filters-out_ch_1

        #skip connect
        if(self._down_rate<1.0):
            self._skip_pool=AdaptAvgPooling(out_hw,name=self._name+"_skip_pool")
        self._skip_conv=ConvBN(out_ch_1,kernel_size=(1,1),use_bn=True,activation=None,name=self._name+"_skip_conv")

        #expand
        self._skip_expands=[]
        for i in range(self._t):
            dconv=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_skip_expand_"+str(i))
            self._skip_expands.append(dconv)
        self._input_expand=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_input_expand")
        if(self._down_rate<1.0):
            self._input_pool=AdaptAvgPooling(out_hw,name=self._name+"_input_pool")
        self._expand_bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_expand_bn")
        self._expand_act=tf.keras.layers.Activation(self._activation,name=self._name+"_expand_act")

        #extract
        self._depth_conv=DepthConvBN(kernel_size=(3,3),strides=(1,1),use_bn=False,activation=None,name=self._name+"_depth_conv")
        self._extract_bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_extract_bn")
        if(self._use_se==True):
            self._sem=SEModule(name=self._name+"_sem")
        self._extract_act=tf.keras.layers.Activation(self._activation,name=self._name+"_extract_act")

        #compress
        self._compress=ConvBN(out_ch_2,kernel_size=(1,1),use_bn=True,activation=None,name=self._name+"_compress")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        self._Build(input_ts)
        
        x=input_ts
        if(self._down_rate<1.0):
            x=self._skip_pool(input_ts)
        p1=self._skip_conv(x)

        skip_expand_x=[]
        for i in range(self._t):
            skip_x=self._skip_expands[i](p1)
            skip_expand_x.append(skip_x)
        skip_expand_x=tf.concat(skip_expand_x,axis=-1)
        input_expand_x=self._input_expand(input_ts)
        if(self._down_rate<1.0):
            input_expand_x=self._input_pool(input_expand_x)
        x=tf.concat([input_expand_x,skip_expand_x],axis=-1)
        x=self._expand_bn(x)
        x=self._expand_act(x)

        x=self._depth_conv(x)
        x=self._extract_bn(x)
        if(self._use_se==True):
            x=self._sem(x)
        x=self._extract_act(x)
        
        p2=self._compress(x)

        output_ts=tf.concat([p1,p2],axis=-1)
        return output_ts

class InputBIFusion(tf.Module):
    def __init__(self,name="inputbufusion"):
        super(InputBIFusion,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,btm_shape,top_shape):
        btm_shape=np.array(btm_shape)
        top_shape=np.array(top_shape)
        target_shape=np.round((btm_shape+top_shape)/2)
        self._btm_down=AdaptAvgPooling(target_shape[0:2],name=self._name+"_btm_down")
        self._top_up=AdaptUpsample(target_shape[0:2],name=self._name+"_top_up")
        self._cslm=CSLModule(filters=top_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_cslm")
    @tf.Module.with_name_scope
    def __call__(self,btm_ts,top_ts):
        btm_shape=btm_ts.get_shape().as_list()[1:]
        top_shape=top_ts.get_shape().as_list()[1:]
        self._Build(btm_shape,top_shape)
        btm_down=self._btm_down(btm_ts)
        top_up=self._top_up(top_ts)
        x=btm_down+top_up
        output_ts=self._cslm(x)
        return output_ts

class CSLFPN(tf.Module):
    def __init__(self,repeat=3,name="cslfpn"):
        super(CSLFPN,self).__init__(name=name)
        self._repeat=repeat
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._fusion_phase1_list=[]
        self._fusion_phase2_list=[]
        for i in range(self._repeat):
            self._fusion_phase1_list.append(FusionPhase1(name=self._name+"_phase1_"+str(i)))
            self._fusion_phase2_list.append(FusionPhase2(name=self._name+"_phase2_"+str(i)))
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        out_ts_list=input_ts_list
        for i in range(self._repeat):
            last_out_ts_list=out_ts_list.copy()
            out_ts_list=self._fusion_phase1_list[i](out_ts_list)
            out_ts_list=self._fusion_phase2_list[i](out_ts_list)
            for ts_idx in range(len(out_ts_list)):
                out_ts_list[ts_idx]=out_ts_list[ts_idx]+last_out_ts_list[ts_idx]
        return out_ts_list

class AdaptAvgPooling(tf.Module):
    def __init__(self,output_hw,name="adaptavgpooling"):
        super(AdaptAvgPooling,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_hw):
        stride=np.floor((input_hw/(self._output_hw)))
        pool_size=input_hw-(self._output_hw-1)*stride
        self._avgpool=tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                       strides=stride,
                                                       name=self._name+"_avgpool")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_hw=input_ts.get_shape().as_list()[1:3]
        self._Build(input_hw)
        output_ts=self._avgpool(input_ts)
        return output_ts

class DepthConvBN(tf.Module):
    def __init__(self,
                 kernel_size,
                 strides=(1,1),
                 padding="same",
                 bias=False,
                 use_bn=True,
                 activation=None,
                 name="depthconvbn"):
        super(DepthConvBN,self).__init__(name=name)
        self._kernel_size=kernel_size
        self._strides=strides
        self._padding=padding
        self._bias=bias
        self._use_bn=use_bn
        self._activation=activation
        self._name=name
        self._Build()
    @tf.Module.with_name_scope
    def _Build(self):
        self._depthconv=tf.keras.layers.DepthwiseConv2D(self._kernel_size,
                                                        self._strides,
                                                        depth_multiplier=1,
                                                        padding=self._padding,
                                                        use_bias=self._bias,
                                                        name=self._name+"_depthconv")
        if(self._use_bn==True):self._bn=tf.keras.layers.BatchNormalization(momentum=0.997,epsilon=1e-4,name=self._name+"_bn")
        self._act=tf.keras.layers.Activation(self._activation,name=self._name+"_act")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        x=self._depthconv(input_ts)
        if(self._use_bn==True):x=self._bn(x)
        x=self._act(x)
        output_ts=x
        return output_ts

class SEModule(tf.Module):
    def __init__(self,name="sem"):
        super(SEModule,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,input_ch):
        self._gap=tf.keras.layers.GlobalAveragePooling2D(name=self._name+"_gap")
        self._conv1=ConvBN(input_ch//4,kernel_size=(1,1),use_bn=False,activation=tf.nn.relu,name=self._name+"_conv1")
        self._conv2=ConvBN(input_ch,kernel_size=(1,1),use_bn=False,activation=hard_sigmoid,name=self._name+"_conv2")
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        input_ch=input_ts.get_shape().as_list()[3]
        self._Build(input_ch)
        x=self._gap(input_ts)
        x=tf.reshape(x,[-1,1,1,input_ch])
        x=self._conv1(x)
        x=self._conv2(x)
        output_ts=input_ts*x
        return output_ts

class AdaptUpsample(tf.Module):
    def __init__(self,output_hw,name="adaptupsample"):
        super(AdaptUpsample,self).__init__(name=name)
        self._output_hw=output_hw
        self._name=name
    @tf.Module.with_name_scope
    def __call__(self,input_ts):
        output_ts=tf.image.resize(input_ts,self._output_hw,method=tf.image.ResizeMethod.BILINEAR)
        return output_ts

class FusionPhase1(tf.Module):
    def __init__(self,name="fusionphase1"):
        super(FusionPhase1,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l2_shape,l4_shape):
        l2_shape=np.array(l2_shape)
        l4_shape=np.array(l4_shape)
        self._l1_down=AdaptAvgPooling(l2_shape[0:2],name=self._name+"_l1_down")
        self._l3_up=AdaptUpsample(l2_shape[0:2],name=self._name+"_l3_up")
        self._l2_cslm=CSLModule(filters=l2_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l2_cslm")
        self._l3_down=AdaptAvgPooling(l4_shape[0:2],name=self._name+"_l3_down")
        self._l5_up=AdaptUpsample(l4_shape[0:2],name=self._name+"_l5_up")
        self._l4_cslm=CSLModule(filters=l4_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l4_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l2_shape=l2.get_shape().as_list()[1:]
        l4_shape=l4.get_shape().as_list()[1:]
        self._Build(l2_shape,l4_shape)

        l1_down=self._l1_down(l1)
        l3_up=self._l3_up(l3)
        
        l2=l2+l1_down+l3_up
        l2=self._l2_cslm(l2)

        l3_down=self._l3_down(l3)
        l5_up=self._l5_up(l5)
        l4=l4+l3_down+l5_up
        l4=self._l4_cslm(l4)
        return [l1,l2,l3,l4,l5]

class FusionPhase2(tf.Module):
    def __init__(self,name="fusionphase2"):
        super(FusionPhase2,self).__init__(name=name)
        self._name=name
    @tf.Module.with_name_scope
    def _Build(self,l1_shape,l3_shape,l5_shape):
        l1_shape=np.array(l1_shape)
        l3_shape=np.array(l3_shape)
        l5_shape=np.array(l5_shape)
        self._l2_up=AdaptUpsample(l1_shape[0:2],name=self._name+"_l2_up")
        self._l1_cslm=CSLModule(filters=l1_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l1_cslm")

        self._l2_down=AdaptAvgPooling(l3_shape[0:2],name=self._name+"_l2_down")
        self._l4_up=AdaptUpsample(l3_shape[0:2],name=self._name+"_l4_up")
        self._l3_cslm=CSLModule(filters=l3_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l3_cslm")

        self._l4_down=AdaptAvgPooling(l5_shape[0:2],name=self._name+"_l4_down")
        self._l5_cslm=CSLModule(filters=l5_shape[2],t=2,use_se=True,activation=mish,name=self._name+"_l5_cslm")
    @tf.Module.with_name_scope
    def __call__(self,input_ts_list):
        l1,l2,l3,l4,l5=input_ts_list
        l1_shape=l1.get_shape().as_list()[1:]
        l3_shape=l3.get_shape().as_list()[1:]
        l5_shape=l5.get_shape().as_list()[1:]
        self._Build(l1_shape,l3_shape,l5_shape)

        l2_up=self._l2_up(l2)
        l1=l1+l2_up
        l1=self._l1_cslm(l1)

        l2_down=self._l2_down(l2)
        l4_up=self._l4_up(l4)
        l3=l3+l2_down+l4_up
        l3=self._l3_cslm(l3)

        l4_down=self._l4_down(l4)
        l5=l5+l4_down
        l5=self._l5_cslm(l5)
        return [l1,l2,l3,l4,l5]