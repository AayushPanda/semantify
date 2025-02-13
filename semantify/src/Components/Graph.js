import React, { useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import data from '../data.json';

// TODO: Make it possible to pan, when the user drags a node try to make it return to its original position slowly...

/*
// Custom node styles
const nodeStyle = {
  background: '#1D70A2',
  width: 20,
  height: 20,
  border: 'none',
  borderRadius: '50%',
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
};

// Define initial nodes and edges outside the component
const initialNodes = [
  {
    id: '1',
    data: { label: '' },
    position: { x: 250, y: 150 },
    style: nodeStyle,
  },
  {
    id: '2',
    data: { label: '' },
    position: { x: 100, y: 300 },
    style: nodeStyle,
  },
  {
    id: '3',
    data: { label: '' },
    position: { x: 400, y: 300 },
    style: nodeStyle,
  },
];

const initialEdges = [
  { 
    id: 'e1-2', 
    source: '1', 
    target: '2',
    style: { stroke: '#1D70A2', strokeWidth: 2 }
  },
  { 
    id: 'e1-3', 
    source: '1', 
    target: '3',
    style: { stroke: '#1D70A2', strokeWidth: 2 }
  },
];
*/
export default function Graph() {
  /*
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );
  */
  const graphRef = useRef();


  useEffect(() => {
    if (graphRef.current && data.nodes.length > 0) {
      // Compute the centroid (average x and y position)
      const avgX =
        data.nodes.reduce((sum, node) => sum + node.x, 0) /
        data.nodes.length;
      const avgY =
        data.nodes.reduce((sum, node) => sum + node.y, 0) /
        data.nodes.length;

      // Center graph at the calculated position
      setTimeout(() => {
        graphRef.current.centerAt(avgX, avgY, 1000);
        graphRef.current.zoom(3, 1000);
      }, 500);

    }
  }, []);
  
  return (
    <div className="graph-container">
      <ForceGraph2D
        ref={graphRef}
        width={window.innerWidth}
        height={window.innerHeight}
        graphData={data}
        nodeAutoColorBy="group"
        backgroundColor="#ffffff"
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.id;
          const fontSize = 12/globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
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
