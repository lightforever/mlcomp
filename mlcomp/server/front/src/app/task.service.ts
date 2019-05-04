import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";

@Injectable({
  providedIn: 'root'
})
export class TaskService extends BaseService{
   protected collection_part: string = 'tasks';
   protected single_part: string = 'task';


}
