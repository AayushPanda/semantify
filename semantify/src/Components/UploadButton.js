export default function UploadButton() {
    return (
        <button className="upload-button" onClick={upload}>
            Upload
        </button>
    )
}

function upload() {
    alert('PRESSED UPLOAD(change feature later)');
}