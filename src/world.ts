import { World } from "./types.ts";
import { Creature } from "./creature.ts";

export const init = () => {
  const canvas = document.getElementById("c");
  const ctx = canvas.getContext("2d");
  const num = 10;
  const fps = 100;

  const world: World = {
    width: document.getElementById("c").width,
    height: document.getElementById("c").height,
    creatures: [],
    context: ctx,
  };

  for (let i = 0; i < num; i++) {
    const x = Math.random() * world.width;
    const y = Math.random() * world.height;
    world.creatures[i] = new Creature(world, x, y);
    world.creatures[i].velocity.random();
  }

  const targetX = (creature: Creature): number => {
    const cohesion = creature.cohesion(world.creatures);
    return cohesion.x / world.width;
  };

  const targetY = (creature: Creature): number => {
    const cohesion = creature.cohesion(world.creatures);
    return cohesion.y / world.height;
  };

  const targetAngle = (creature: Creature): number => {
    const alignment = creature.align(world.creatures);
    return (alignment.angle + Math.PI) / (Math.PI * 2);
  };

  const loop = (): void => {
    ctx.clearRect(0, 0, canvas.width, canvas.height); 
    ctx.fillStyle,ctx.strokeStyle = 'white';
    ctx.stroke();

    const creatures = world.creatures;

    creatures.forEach((creature: Creature) => {
      const input = [];
      for (const i in creatures) {
        input.push(creatures[i].location.x);
        input.push(creatures[i].location.y);
        input.push(creatures[i].velocity.x);
        input.push(creatures[i].velocity.y);
      }

      const output = creature.network.activate(input);
      creature.moveTo(output);

      const learningRate = 0.3;
      const target = [
        targetX(creature),
        targetY(creature),
        targetAngle(creature),
      ];

      creature.network.propagate(learningRate, target);
      creature.draw();
    });
    setTimeout(loop, 1000 / fps);
  };

  loop();

};
