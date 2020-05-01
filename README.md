# geometric_processing_net

Neural network architecture for training on mesh or vertes set of a mesh. 
The architecture is based on [PointNet](http://stanford.edu/~rqi/pointnet/).

The framework is used to estimate the Hessian of symmetric dirichlet deformation energy from vertes displacments. 
Estimating the Hessian might be useful in geometric processing to run defomration minimization. 

Inputs to the network (listed here to give a feeling of what the input looks like, at this point no drawing is provided, though the drawing would be very helpful to see the network architecture)

- The vertex set of the mesh (variance normalized source vertices)
- Displacment of each vertex from variance normalized source to variance normalized target
- A single edge, given by pair of source vertices i and j to compute w_ij. Where w_ij are used to aseemble the Hessian.
Where hessian is assumed to have the form of graph Laplacian where the weights between edges i and j are w_ij. In principle, to get the full Laplacian 
the user has to run the network multiple times, providing each time a different pair ij to get the desired w_ij. But current implementation
allows to generate architecture with shared weights that will calculate w_ij for many edges in parallel. 


Outputs

- w_ij used to assemble Hessian. Where the Hessian is assumed to have graph Laplacian like structure (see `assemble_h.py` to see how Hessian is assembled form w_ij)


# Running the framework 

The frameworks provides tools to generated data for network training for visualization and various axilluary geometric processing function to run energy calcualtion and derivations as well as some optimization tools to compare results with basis non convex optimization schemes. To get started run `network_test.py` this runs the provided pretrained network to produce second order prediction for the energy around a few random deformations. Then the second order approximation is compared with actual  values of the energy along a certain direction (in the high-dimensional space of the deformations).  

![input example](https://github.com/sgregnt/geometric_processing_net/blob/master/pics/mesh_with_displacments.png "Vertex set on 2D mesh with its displacment")


