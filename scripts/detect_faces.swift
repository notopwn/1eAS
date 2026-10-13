import Foundation
import Vision

struct Face: Codable {
    let x: Double; let y: Double; let width: Double; let height: Double; let confidence: Double
}
struct Detection: Codable {
    let path: String; let faces: [Face]; let error: String?
}
let paths = try JSONDecoder().decode([String].self, from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1])))
var results: [Detection] = []
for (index, path) in paths.enumerated() {
    let result: Detection = autoreleasepool {
        do {
            let request = VNDetectFaceRectanglesRequest()
            let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path), options: [:])
            try handler.perform([request])
            let faces = (request.results ?? []).map { face in
                let box = face.boundingBox
                return Face(x: box.origin.x, y: box.origin.y, width: box.width, height: box.height, confidence: Double(face.confidence))
            }
            return Detection(path: path, faces: faces, error: nil)
        } catch {
            return Detection(path: path, faces: [], error: String(describing: error))
        }
    }
    results.append(result)
    if (index + 1) % 50 == 0 {
        FileHandle.standardError.write(Data("Face detection: \(index + 1)/\(paths.count)\n".utf8))
    }
}
FileHandle.standardOutput.write(try JSONEncoder().encode(results))
