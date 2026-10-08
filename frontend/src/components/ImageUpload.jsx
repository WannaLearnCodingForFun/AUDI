import { useRef, useState } from 'react'

export default function ImageUpload({ onUpload, loading }) {
  const inputRef = useRef()
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState('')

  const handleFile = (file) => {
    if (!file) return

    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp']
    const ext = file.name.split('.').pop()?.toLowerCase()
    const validExts = ['jpg', 'jpeg', 'png', 'webp', 'gif', 'bmp', 'heic']

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
      onUpload(file, { clientError: `Unsupported file: ${file.name}. Use JPEG or PNG.` })
      return
    }

    setFileName(file.name)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    onUpload(file)
  }

  const handleInput = (e) => handleFile(e.target.files?.[0])
  const handleDrop = (e) => {
    e.preventDefault()
    handleFile(e.dataTransfer.files?.[0])
  }

  return (
    <div>
      <div
        className="upload-zone"
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input ref={inputRef} type="file" accept="image/*,.jpg,.jpeg,.png,.webp" hidden onChange={handleInput} />
        {loading ? (
          <p>Analyzing...</p>
        ) : preview ? (
          <>
            <img src={preview} alt="Upload preview" style={{ maxHeight: 200, borderRadius: 8, marginBottom: '0.75rem' }} />
            <p style={{ fontSize: '0.875rem', color: 'var(--muted)' }}>{fileName}</p>
            <p style={{ fontSize: '0.75rem', color: 'var(--accent2)', marginTop: '0.5rem' }}>Click to change image</p>
          </>
        ) : (
          <>
            <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>Upload Inspection Image</p>
            <p style={{ color: 'var(--muted)', fontSize: '0.875rem' }}>Drop image here or click to browse (JPEG, PNG)</p>
          </>
        )}
      </div>
    </div>
  )
}
