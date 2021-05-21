import { Creature } from "./creature.ts";

export type World = {
  width: number;
  height: number;
  creatures: Creature[] | [];
  context: any;
};
