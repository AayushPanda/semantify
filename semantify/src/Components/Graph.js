import React, { useCallback } from 'react';
import ReactFlow, {
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
} from 'reactflow';
import 'reactflow/dist/style.css';

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

export default function Graph() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        fitView
        style={{ background: '#E1E1E1', borderRadius: '15px' }}
        nodesDraggable={true}
        nodesConnectable={false}
        elementsSelectable={true}
      >
        <Background color="#D0D0D0" gap={20} size={1} />
        <Controls 
          style={{
            button: {
              backgroundColor: 'white',
              color: '#1D70A2',
              borderRadius: '4px',
            },
          }}
        />
      </ReactFlow>
    </div>
  );
}
