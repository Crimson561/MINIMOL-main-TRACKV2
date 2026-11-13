import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";


const app = express()
const port = 3000
const httpServer = createServer(app)
const io = new Server(httpServer , {})

app.get("/" , (req, res) => {
    res.send("Hello World")
});

io.on("connection", (socket) => {
  console.log(`client connect hogya oyeeeeeeeeee: ${socket.id}`);
});

httpServer.listen(port , () => {
    console.log(`trying to make this express server work on ${port}`)
});

