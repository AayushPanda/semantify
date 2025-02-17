import React from 'react';
import axios from 'axios';

export default function UploadButton() {
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            axios.post('http://localhost:8000/upload', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            })
            .then(response => {
                console.log('Success:', response.data);
                // Update UI with response data
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    };

    const handleButtonClick = () => {
        document.getElementById('upload-input').click();
    };

    return (
        <>
            <input
                type="file"
                accept=".zip"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="upload-input"
            />
            <button className="upload-button" onClick={handleButtonClick}>
                Upload
            </button>
        </>
    );
}