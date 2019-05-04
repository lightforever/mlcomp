import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";

@Injectable({
  providedIn: 'root'
})
export class ProjectService extends BaseService{
   protected collection_part: string = 'projects';
   protected single_part: string = 'project';


}