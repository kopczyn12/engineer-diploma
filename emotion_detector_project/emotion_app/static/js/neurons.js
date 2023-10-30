const canvas = document.getElementById('neuronCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const neurons = [];
const totalNeurons = 800;

class Neuron {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.size = Math.random() * 5 + 1;
        this.speedX = Math.random() * 3 - 1.5;
        this.speedY = Math.random() * 3 - 1.5;
    }
    
    update() {
        this.x += this.speedX;
        this.y += this.speedY;

        if (this.size > 0.2) this.size -= 0.1;

        if (this.x > canvas.width) {
            this.x = 0;
        } else if (this.x < 0) {
            this.x = canvas.width;
        }

        if (this.y > canvas.height) {
            this.y = 0;
        } else if (this.y < 0) {
            this.y = canvas.height;
        }
    }
    
    draw() {
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fill();
    }
}

let tick = 0;

function init() {

    for (let i = 0; i < 330; i++) {
        const x = Math.random() * canvas.width;
        const y = Math.random() * canvas.height;
        neurons.push(new Neuron(x, y));
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (tick % 50 === 0) {
        const x = Math.random() * canvas.width;
        const y = 0; 
        neurons.push(new Neuron(x, y));
    }
    tick++;

    for (let i = 0; i < neurons.length; i++) {
        neurons[i].update();
        neurons[i].draw();
        
        for (let j = i; j < neurons.length; j++) {
            const dx = neurons[i].x - neurons[j].x;
            const dy = neurons[i].y - neurons[j].y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < 100) {
                ctx.beginPath();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 0.2;
                ctx.moveTo(neurons[i].x, neurons[i].y);
                ctx.lineTo(neurons[j].x, neurons[j].y);
                ctx.stroke();
                ctx.closePath();
                
                ctx.beginPath();
                ctx.arc(neurons[i].x, neurons[i].y, 2.5, 0, Math.PI * 2); // Dot for neuron i
                ctx.fill();
                ctx.closePath();
            
                ctx.beginPath();
                ctx.arc(neurons[j].x, neurons[j].y, 2.5, 0, Math.PI * 2); // Dot for neuron j
                ctx.fill();
                ctx.closePath();
            }
        }
    }

    if (neurons.length > totalNeurons) {
        neurons.splice(0, 20);
    }

    requestAnimationFrame(animate);
}
init();
animate();
