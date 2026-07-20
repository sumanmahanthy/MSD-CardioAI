import React from "react";

function PreprocessingViewer({original,processed}){

if(!original || !processed){

return(

<div className="card p-4 shadow-sm border-0">

<h5>Preprocessing Visualization</h5>

<p>No preprocessing data available</p>

</div>

)

}

return(

<div className="card p-4 shadow-sm border-0">

<h5>Image Preprocessing</h5>

<div className="row">

<div className="col-md-6">

<h6>Original X-ray</h6>

<img src={original} className="img-fluid"/>

</div>

<div className="col-md-6">

<h6>CLAHE Enhanced</h6>

<img src={processed} className="img-fluid"/>

</div>

</div>

</div>

)

}

export default PreprocessingViewer