import { InMemoryDbService } from 'angular-in-memory-web-api';
import { Project } from './models';
import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class InMemoryDataService implements InMemoryDbService {
  createDb() {
    const projects = [
      {id: 1, name: 'first project', 'lastActivity': new Date(), tasks: [
         {id: 1, name: 'first task', project: 1},
         {id: 2, name: 'second task', project: 1},
      ]},
      {id: 2, name: 'second project', 'lastActivity': new Date(), tasks: [
         {id: 3, name: 'third task', project: 2}
      ]},
    ];
    const tasks = [
      {id: 1, name: 'first task', project: 1},
      {id: 2, name: 'second task', project: 1},
      {id: 3, name: 'third task', project: 2},
    ];
    return {projects, tasks};
  }

}
