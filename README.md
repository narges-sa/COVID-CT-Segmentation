# COVID TV-UNet: Segmenting COVID-19 Chest CT Images Using Connectivity Imposed U-Net

The main contributions of our work can be summarized as
follows:

• Development of of a novel connectivity-promoting
regularization loss function for an image segmentation
framework detecting pathologic COVID-19 regions in
pulmonary CT images.

• Quantitative validation showing improved performance attributable to our new TV-UNet approach
compared to published state-of-the-art segmentation
approaches.


# COVID CT SEGMENTATION DATASET
We have combined all available data from the COVID-19
CT segmentation dataset [28], consisting of 929 CT slices
from 49 patients. Out of these, 473 CT-image slices are labeled as including COVID-19 pathologies with Ground-Glass
pathology regions identified by expert tracing. The remaining
456 CT image slices are labeled as COVID-19 pathologyfree. CT-slice sizes were either 512×512 or 630×630. While
a small subset of the 929 CT images also have regions of
additional pathologies identified and labeled as Consolidation
and/or Pleural Effusion, these pathologies were not consistently available for all the data. After consulting with a boardcertified radiologist (GS) who confirmed that Ground-Glass pathology is most relevant for detecting COVID-19, this work
focuses on the Ground Glass mask, and does not consider the
Consolidation and Pleural Effusion masks due to their small
numbers and lack of consistency across the dataset.
datasets [here](http://medicalsegmentation.com/covid19/)

One such dataset with semi-supervised COVID-19
segmentations (COVID-SemiSeg) was recently reported in
[17]. The COVID-SemiSeg dataset consists of two sets. The first one contains 1600 pseudo labels generated by Semi-InfNet model and 50 labels by expert physicians. The second set
includes 50 multi-class labels. Overall, there are 48 images
that can be used for performance-comparison assessment and
these CT data were used to compare our TV-Unet approach
with other methods.





![alt tag](https://github.com/narges-sa/COVID-CT-Segmentation/blob/readme-changes/results/normal%26COVID.jpg? )
<p align="center">
  <img src="Fig. 1. The difference between normal and COVID-19 images" width="350" alt="Fig. 1. The difference between normal and COVID-19 images">
</p>



![alt text](https://github.com/narges-sa/COVID-CT-Segmentation/blob/readme-changes/results/COVID.jpg)

Fig. 2. Sample images from the COVID-19 CT segmentation dataset. The
first row shows two COVID-19 images. The red boundary contours in the
second row denote regions of COVID-19 Ground-Glass pathology and are not
a part of the original image data.The third row shows Ground-Glass masks

# Visualization Results:
![alt text](https://github.com/narges-sa/COVID-CT-Segmentation/blob/readme-changes/results/maskB%26TV.jpg)
Fig. 3. Predicted segmentation masks by U-Net trained from scratch and the
proposed TV-UNet for a typical sample images from the testing set.

# Usage Right:

This work is done by Narges Saeedizadeh, Shervin Minaee, Rahele Kafieh, Shakib Yazdani, and Milan Sonka (the previous editor in chief of IEEE TMI). 

The Arxiv version of the paper can be downloaded from [here](https://arxiv.org/pdf/2007.12303.pdf). 

If you find this work useful, you can refer our work as:

@article{
  title={COVID TV-UNet: Segmenting COVID-19 Chest CT Images Using Connectivity Imposed U-Net},
  author={Saeedizadeh, Narges and Minaee, Shervin and Kafieh, Rahele and Yazdani, Shakib and Sonka, Milan},
  journal={arXiv preprint arXiv:2007.12303},
  year={2020}
}
