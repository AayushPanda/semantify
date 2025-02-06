
export default function DownloadButton() {
    return(
        <button onClick={download}>Download</button>
    )
}

function download() {
    alert('PRESSED DOWNLOAD');
}