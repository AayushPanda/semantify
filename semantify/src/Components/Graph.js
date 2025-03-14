import React, { useRef, useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import data_subheadings from '../data_subheadings.json';
import data_headings from '../data_headings.json';


const calculateCenter = (nodes) => {
  if (!nodes || nodes.length === 0) return { x: 0, y: 0 };
  const avgX = nodes.reduce((sum, node) => sum + node.x, 0) / nodes.length;
  const avgY = nodes.reduce((sum, node) => sum + node.y, 0) / nodes.length;
  return { x: avgX, y: avgY };
};

const findNode = (nodes, idName) => { // WILL GET THE AVERAGE X AND Y OF WHAT YOU WANT...
  if (!nodes || nodes.length === 0) return { x: 0, y: 0 };
  const avgX = nodes.filter(node => node.id=idName).reduce((sum, node) => sum + node.x, 0) / nodes.length;
  const avgY = nodes.filter(node => node.Inner==idName).reduce((sum, node) => sum + node.y, 0) / nodes.length;
  return { x: avgX, y: avgY };
}


// HAVE AN ARRAY FOR HEADINGS....
// I SHOULD GET A PARAMETER TO ADD SMTH TO THE DATA....
// ALSO NEED A METHOD TO GO TO SOMEWHERE AND ZOOM IN ON A CERTAIN PART...
// NOW HOW TO MAKE METHODS THAT WILL UPDATE ON EVERY CALL?
// Should do a loop to find all the different parameter thingies...
let start = true;


export default function Graph() {

  const graphRef = useRef();

  const [zoomLevel, setZoomLevel] = useState(null);
  const [dataFile, setDataFile] = useState(data_headings); // Default file

  // Sets and updates zoomLevel
  useEffect(() => { 
    if (!graphRef.current) return;

    let animationFrameId;

    const trackZoom = () => {
      const newZoom = graphRef.current.zoom();
      if (Math.abs(newZoom-zoomLevel)!=0) {
        setZoomLevel(newZoom);
        console.log("Zoom Level Changed:", newZoom);
        if(newZoom < 2){
          setDataFile(data_subheadings);
        }
        else{
          setDataFile(data_headings);
        }
      }
      animationFrameId = requestAnimationFrame(trackZoom);
    };
    setTimeout(50);

    trackZoom();

    return () => cancelAnimationFrame(animationFrameId);
  }, [zoomLevel]);



  return (
    <div className="graph-container">
      <ForceGraph2D
        ref={graphRef}
        width={window.innerWidth}
        height={window.innerHeight}
        graphData={dataFile}
        nodeAutoColorBy="group"

        cooldownTicks={0}
        yOffset={0}
        enableNodeDrag={false}
        onEngineStop={() => {
          if (graphRef.current) {
            const { x, y } = calculateCenter(dataFile.nodes);
            if(start){
              graphRef.current.centerAt(x, y, 800);
              start = false;
            }
          }
        }}
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.id;
          const fontSize = (node.group==0) ? 30/globalScale : 12/globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

          ctx.fillStyle = 'rgba(255, 255, 255, 0)';
          ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = node.color;
          ctx.fillText(label, node.x, node.y);

          node.__bckgDimensions = bckgDimensions;
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          const bckgDimensions = node.__bckgDimensions;
          bckgDimensions && ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);
        }}
        enableZoom={true}
        enablePan={true}

      />
    </div>
  );
}
