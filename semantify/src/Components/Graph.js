import React, { useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './test.json'
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
  const data = {
    "nodes": [
      {"id": "Myriel", "group": 1},
      {"id": "Napoleon", "group": 100},
      {"id": "Mlle.Baptistine", "group": 1},
      {"id": "Mme.Magloire", "group": 1},
      {"id": "CountessdeLo", "group": 1},
      {"id": "Geborand", "group": 1},
      {"id": "Champtercier", "group": 1},
      {"id": "Cravatte", "group": 1},
      {"id": "Count", "group": 1},
      {"id": "OldMan", "group": 1},
      {"id": "Labarre", "group": 2},
      {"id": "Valjean", "group": 2},
      {"id": "Marguerite", "group": 3},
      {"id": "Mme.deR", "group": 2},
      {"id": "Isabeau", "group": 2},
      {"id": "Gervais", "group": 2},
      {"id": "Tholomyes", "group": 3},
      {"id": "Listolier", "group": 3},
      {"id": "Fameuil", "group": 3},
      {"id": "Blacheville", "group": 3},
      {"id": "Favourite", "group": 3},
      {"id": "Dahlia", "group": 3},
      {"id": "Zephine", "group": 3},
      {"id": "Fantine", "group": 3},
      {"id": "Mme.Thenardier", "group": 4},
      {"id": "Thenardier", "group": 4},
      {"id": "Cosette", "group": 5},
      {"id": "Javert", "group": 4},
      {"id": "Fauchelevent", "group": 0},
      {"id": "Bamatabois", "group": 2},
      {"id": "Perpetue", "group": 3},
      {"id": "Simplice", "group": 2},
      {"id": "Scaufflaire", "group": 2},
      {"id": "Woman1", "group": 2},
      {"id": "Judge", "group": 2},
      {"id": "Champmathieu", "group": 2},
      {"id": "Brevet", "group": 2},
      {"id": "Chenildieu", "group": 2},
      {"id": "Cochepaille", "group": 2},
      {"id": "Pontmercy", "group": 4},
      {"id": "Boulatruelle", "group": 6},
      {"id": "Eponine", "group": 4},
      {"id": "Anzelma", "group": 4},
      {"id": "Woman2", "group": 5},
      {"id": "MotherInnocent", "group": 0},
      {"id": "Gribier", "group": 0},
      {"id": "Jondrette", "group": 7},
      {"id": "Mme.Burgon", "group": 7},
      {"id": "Gavroche", "group": 8},
      {"id": "Gillenormand", "group": 5},
      {"id": "Magnon", "group": 5},
      {"id": "Mlle.Gillenormand", "group": 5},
      {"id": "Mme.Pontmercy", "group": 5},
      {"id": "Mlle.Vaubois", "group": 5},
      {"id": "Lt.Gillenormand", "group": 5},
      {"id": "Marius", "group": 8},
      {"id": "BaronessT", "group": 5},
      {"id": "Mabeuf", "group": 8},
      {"id": "Enjolras", "group": 8},
      {"id": "Combeferre", "group": 8},
      {"id": "Prouvaire", "group": 8},
      {"id": "Feuilly", "group": 8},
      {"id": "Courfeyrac", "group": 8},
      {"id": "Bahorel", "group": 8},
      {"id": "Bossuet", "group": 8},
      {"id": "Joly", "group": 8},
      {"id": "Grantaire", "group": 8},
      {"id": "MotherPlutarch", "group": 9},
      {"id": "Gueulemer", "group": 4},
      {"id": "Babet", "group": 4},
      {"id": "Claquesous", "group": 4},
      {"id": "Montparnasse", "group": 4},
      {"id": "Toussaint", "group": 5},
      {"id": "Child1", "group": 10},
      {"id": "Child2", "group": 10},
      {"id": "Brujon", "group": 4},
      {"id": "Mme.Hucheloup", "group": 8}
    ],
    "links": [
      {"source": "Napoleon", "target": "Myriel", "value": 3},
      {"source": "Napoleon", "target": "Myriel", "value": 3},
      {"source": "Napoleon", "target": "Myriel", "value": 3},
      {"source": "Napoleon", "target": "Myriel", "value": 3},

      {"source": "Napoleon", "target": "Mme.Magloire", "value": 3},
      {"source": "Mme.Magloire", "target": "Mme.Hucheloup", "value": 3},
      {"source": "Mme.Hucheloup", "target": "CountessdeLo", "value": 1000}
    ]
  }

  return (

    
    
    <div style={{ width: '100%', height: '100%' }}>
      <ForceGraph2D
        width={2030} // CHANGE THESE AFTTER TO BE MORE DYNAMIC
        height={1000}
        graphData={data}
        nodeAutoColorBy="group"
        backgroundColor="#ffffff"
        nodeCanvasObject={(node, ctx, globalScale) => {
          const label = node.id;
          const fontSize = 12/globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          const textWidth = ctx.measureText(label).width;
          const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2); // some padding

          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);

          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = node.color;
          ctx.fillText(label, node.x, node.y);

          node.__bckgDimensions = bckgDimensions; // to re-use in nodePointerAreaPaint
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          const bckgDimensions = node.__bckgDimensions;
          bckgDimensions && ctx.fillRect(node.x - bckgDimensions[0] / 2, node.y - bckgDimensions[1] / 2, ...bckgDimensions);
        }}
      />

      {/*
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
      */}
    </div>
    
  );
}
