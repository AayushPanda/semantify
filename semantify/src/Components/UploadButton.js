
export default function UploadButton() {
    return(
        <button onClick={upload}>Upload</button>
    )
}

function upload() {
    alert('PRESSED UPLOAD');
}