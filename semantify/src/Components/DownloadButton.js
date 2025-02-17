export default function DownloadButton() {
    return (
        <button className="download-button" onClick={download}>
            Download
        </button>
    )
}

function download() {
    alert('Pressed Download, (add in features later)');
}