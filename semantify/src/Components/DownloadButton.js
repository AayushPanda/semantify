import React from 'react';
import axios from 'axios';

export default function DownloadButton() {
    const handleDownload = async () => {
        try {
            const response = await axios.get('http://localhost:8000/download', { // get request to download from endpoint
                responseType: 'blob', // handle only binary data to allow browser to treat as downloadble folder
            });

            // Creates a URL for blob
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'organised_files.zip'); // Set the file name

            document.body.appendChild(link);
            link.click();

            link.parentNode.removeChild(link);
        } catch (error) {
            console.error('Error downloading the file:', error);
        }
    };

    return (
        <button className="download-button" onClick={handleDownload}>
            Download
        </button>
    );
}