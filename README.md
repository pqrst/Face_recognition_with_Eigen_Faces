# Face_recognition_with_Eigen_Faces

An approach to the detection and identification of human faces has been presented and described for a face recognition system that identifies a person by comparing characteristics of the face to those of individuals in the training dataset. In this approach, face images are projected into a feature space that best encodes the variation among known face images. The face space is defined by the “Eigen faces” which are eigenvectors of the set of faces. 
The Eigenfaces method takes a holistic approach to face recognition: A facial image is a set of points having a high-dimension. Hence, a lower-dimensional representation is found where classiﬁcation becomes easy. The lower-dimensional subspace is found with Principal Component Analysis, which identiﬁes the axes with maximum variance. While this kind of transformation is optimal from a reconstruction standpoint, it doesn’t take any class labels into account. The basic idea for this approach is to minimize the variance within a class, while maximizing the variance between the classes at the same time. 
