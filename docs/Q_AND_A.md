# Likely examiner questions

**Q. How many bones of the human body can you test?**  
A. The adult skeleton has **206** bones. We do **not** have 206 output classes. We cover **11 skeletal regions** (~**42** named bones/groups) and **10** fracture morphologies. The published YOLO model uses **7 upper-limb** classes. Femur, knee, ankle, spine, pelvis are in the software map and need extra labelled images for a strong detector.

**Q. Can you test a skull or rib fracture?**  
A. Not as a dedicated class. Those would need new labels and data.

**Q. Is 20,000 images 20,000 of each type?**  
A. No. GRAZPEDWRI-DX is **20,327 pediatric wrist** radiographs. The 10-type paper set is **1,129**. Together with YOLO + FracAtlas you *scale* toward 20k+ films, not 20k per class.

**Q. Why VGG-16 if it is old?**  
A. The IEEE Access 2025 study found VGG-16 and VGG-16+RF strongest (**95%**) on this 10-class X-ray set; EfficientNetB0+XGBoost lagged (~**41%**). We keep that comparison.

**Q. What is mAP50?**  
A. Mean Average Precision with IoU threshold 0.50 for detection boxes. Abstract: **86%** on the 7-class YOLO set.

**Q. Is the metal plate on the X-ray real surgery?**  
A. No. OpenCV overlay for teaching alignment/fixation concepts only.

**Q. How do you stop anyone retraining production models?**  
A. Role-based access: only **admin** reaches `/admin` and `/api/admin/retrain`. Clinicians use upload/inference only. Passwords are hashed; actions go to the audit log.

**Q. Grad-CAM?**  
A. Heatmap from the last VGG convolution (`block5_conv3`) showing which pixels drove the class. If the CNN is not loaded, we show an edge/intensity saliency map and say so.

**Q. Greenstick in adults?**  
A. Greenstick is typically paediatric. The class exists because it is in the paper taxonomy; prevalence in an adult trauma set will be low.

**Q. DICOM?**  
A. `.dcm` is accepted via pydicom and converted to an 8-bit image for the same pipeline.
