# Khung Xương 2D

Web app giúp người không chuyên làm game: tải **một ảnh nhân vật nhìn nghiêng** (vẽ bằng AI cũng được),
app tự dựng khung xương, tách các phần cơ thể, rồi tạo chuyển động (đi, chạy, đứng thở) và xuất spritesheet.
Mọi thứ chạy ngay trong trình duyệt, kể cả AI, không cần card đồ họa rời và không gửi ảnh đi đâu.

## Các bước trong app

1. **Ảnh**: kéo thả ảnh. Nền trơn được tách ngay; nền phức tạp thì bấm "Dùng AI tách nền".
2. **Khớp xương**: app tự đoán dáng người (người lớn nam/nữ, trẻ em, chibi), hướng nhìn và vị trí 14 khớp. Kéo chấm để sửa.
3. **Tách lớp**: AI (MobileSAM) tách đầu, thân, tay trước/sau, chân trước/sau, vũ khí. Bấm chuột trái/phải để sửa.
   Thêm lớp chi tiết tuỳ chọn: tóc sau, tóc mái, mặt, mắt + mi, miệng, áo, phụ kiện, quần, giày.
4. **Chuyển động**: lưới biến dạng có trọng số xương (không cắt cứng từng mảnh nên không có vệt trắng ở khớp),
   ma-nơ-canh lót dưới quần áo che khe hở, thân được tô lấp chỗ bị tay che, tay/chân sau nhân bản từ tay/chân trước.
5. **Xuất file**: spritesheet PNG (mỗi hành động một hàng) + JSON (khung hình, thời lượng, fps) dùng cho Phaser, Pixi, Godot, Unity…

**3D Ma-nơ-canh**: 4 ma-nơ-canh 3D (nam, nữ, trẻ em, chibi) có khung xương, xoay 360°, chọn góc camera
và xem ngay ảnh 2D + chuyển động ở góc đó; tải spritesheet của góc đang chọn.

## Chạy trên máy

```bash
npm install
npm run dev      # mở http://localhost:5173
npm run build    # bản tĩnh trong dist/
```

## Ghi chú kỹ thuật

- TypeScript + Vite, không framework. WebGL2 cho lưới biến dạng 2D, three.js cho phần 3D.
- AI: MobileSAM (ONNX, ~37 MB, tải một lần rồi lưu cache) chạy bằng onnxruntime-web trong Web Worker:
  thử WebGPU trước, không được thì dùng CPU (WebAssembly, đa luồng khi trình duyệt cho phép).
  Trên CPU 4 nhân: phân tích ảnh ~3–4 giây, tách lớp ~3 giây.
- File mô hình lấy từ bản phát hành của X-AnyLabeling (`v0.2.0`), xuất từ MobileSAM (Apache-2.0).
  Nên tự xuất lại từ trọng số gốc trước khi phát hành thương mại.
- `public/samples/` là 4 ảnh thử từ bộ ảnh mẫu của dự án.
