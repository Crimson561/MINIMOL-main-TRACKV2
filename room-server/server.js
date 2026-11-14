import express from "express";
import { createServer } from "http";
import { Server } from "socket.io";


const app = express()
const port = 3000
const httpServer = createServer(app)
const io = new Server(httpServer, {
  cors: {
    origin: "http://127.0.0.1:5500",
  },
});

app.get("/" , (req, res) => {
    res.send("Hello World")
});

io.on("connection", (socket) => {
  console.log(`client connect hogya oyeeeeeeeeee: ${socket.id}`);

socket.on("ping-server", (data) => {
  console.log(`ping server recieve hogya!! `, data);

    socket.broadcast.emit("server-reply" , {
    message: data , 
    timestamp: Date.now()
  });
});

});


httpServer.listen(port , () => {
    console.log(`trying to make this express server work on ${port}`)
});

