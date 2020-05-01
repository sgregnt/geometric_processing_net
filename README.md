# geometric_processing_net

Neural network architecture for training on mesh or vertes set of a mesh. 
The architecture is based on [PointNet](http://stanford.edu/~rqi/pointnet/).

The framework is used to estimate the Hessian of symmetric dirichlet deformation energy from vertes displacments. 
Estimating the Hessian might be useful in geometric processing to run defomration minimization. 

Inputs to the network (listed here to give a feeling of what the input looks like, at this point no drawing is provided, though the drawing would be very helpful to see the network architecture)

- The vertex set of the mesh (variance normalized source vertices)
- Displacment of each vertex from variance normalized source to variance normalized target
- A single edge, the ij vertix, to compute w_ij, specified for the source vertices. 
Hessian is assumed to have the form of graph Laplacian where the weights between edges are w_ij. In principle, to get the full laplacian 
the user has to run the network multiple times, providing each time a different pair ij to get the desired w_ij. But current implementation
allows to generate architecture with shared weights that will calculate w_ij for many edges in parallel. 


Outputs

- w_ij used to assemble Hessian. Where the Hessian is assumed to have graph Laplacian like structure (see `assemble_h.py` to see how Hessian is assembled form w_ij)

